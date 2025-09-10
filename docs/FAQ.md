# Frequently Asked Questions (FAQ)

This document answers common questions about the AI Job Queue System, uv package management, and worker configuration.

## Table of Contents

- [General Questions](#general-questions)
- [uv Package Management](#uv-package-management)
- [Worker Management](#worker-management)
- [API Usage](#api-usage)
- [Troubleshooting](#troubleshooting)
- [Deployment](#deployment)

## General Questions

### Q: What is this system for?

**A:** This is a microservices-based AI job queue system that separates the FastAPI backend from AI training/inference workers. It allows you to:

- Submit training and inference jobs via REST API
- Run multiple AI workers with different ML frameworks (PyTorch, TensorFlow, Scikit-learn)
- Scale workers independently based on workload
- Use GPU acceleration for compute-intensive tasks
- Manage dependencies independently for each service

### Q: How is this different from running ML models directly?

**A:** This system provides:

- **Scalability**: Multiple workers can process jobs in parallel
- **Framework Flexibility**: Different workers can use different ML frameworks
- **Resource Management**: GPU memory management and job queuing
- **API Interface**: Standard REST API for job submission and monitoring
- **Dependency Isolation**: Backend and workers have separate dependencies
- **Production Ready**: Docker support, health checks, and monitoring

### Q: What ML frameworks are supported?

**A:** The system supports:

- **PyTorch 2.0** (CPU and GPU with CUDA 11.8)
- **PyTorch 2.1** (CPU and GPU with CUDA 12.1)
- **TensorFlow 2.13** (CPU)
- **Scikit-learn 1.3** (CPU)

## uv Package Management

### Q: How does uv handle multiple virtual environments in the same directory?

**A:** uv uses a **single virtual environment** (`.venv`) per project directory, but manages dependencies selectively through "extras":

- Only one `.venv` directory exists per project
- `uv sync --extra pytorch_2_1` installs only PyTorch 2.1 dependencies
- `uv sync --extra tensorflow` replaces PyTorch with TensorFlow dependencies
- The lock file (`uv.lock`) reflects the current state, not a history

### Q: After running `uv sync --extra pytorch_2_0` then `uv sync --extra pytorch_2_1`, what does uv.lock contain?

**A:** The `uv.lock` file will contain **PyTorch 2.1.0** (the last installed version).

**Why:** uv replaces dependencies when switching extras. The lock file always reflects the current state of the virtual environment, not a history of installations.

### Q: What dependencies does the `.venv` directory have after switching extras?

**A:** The `.venv` directory will contain **PyTorch 2.1** dependencies (the last installed version).

**What's installed:**
- Base dependencies (redis, numpy, pandas, psutil)
- PyTorch 2.1.0, torchvision 0.16.0, torchaudio 2.1.0
- PyTorch 2.1 ecosystem packages (transformers, datasets, accelerate)

**What's removed:**
- PyTorch 2.0.1, torchvision 0.15.2, torchaudio 2.0.2

### Q: How does a running worker process handle dependency changes?

**A:** A running worker process **continues using the version it started with**, even after you change dependencies.

**Example:**
```bash
# Terminal 1: Start worker with PyTorch 2.0
uv sync --extra pytorch_2_0
uv run python src/worker.py  # Worker loads PyTorch 2.0 into memory

# Terminal 2: Switch to PyTorch 2.1
uv sync --extra pytorch_2_1

# Result: Worker in Terminal 1 still uses PyTorch 2.0 in memory
# New processes will use PyTorch 2.1
```

**To use new dependencies:** Stop and restart the worker process.

### Q: Is `uv run` different from Poetry regarding in-memory packages?

**A:** **No, `uv run` is not different from Poetry regarding in-memory packages.** Both have identical behavior.

**The Truth:** Both `uv run` and Poetry have the same in-memory behavior - this is a Python process characteristic, not a package manager feature.

**How Both Work:**
```bash
# Poetry approach
poetry install --extras pytorch_2_0
poetry run python src/worker.py  # Starts Python process with PyTorch 2.0 in memory
poetry install --extras pytorch_2_1  # Changes .venv to PyTorch 2.1
# Running worker still has PyTorch 2.0 in memory

# uv approach  
uv sync --extra pytorch_2_0
uv run python src/worker.py  # Starts Python process with PyTorch 2.0 in memory
uv sync --extra pytorch_2_1  # Changes .venv to PyTorch 2.1
# Running worker still has PyTorch 2.0 in memory
```

**Why This Happens:**
- **Python imports modules once** and keeps them in memory
- **Virtual environment changes** don't affect running processes
- **New processes** use the current virtual environment state
- **Both Poetry and uv** create Python processes the same way

**The Real Differences:**
| Aspect | Poetry | uv |
|--------|--------|-----|
| **In-memory behavior** | Same as uv | Same as Poetry |
| **Dependency installation** | All extras installed | Selective installation |
| **Virtual environment** | Single `.venv` | Single `.venv` |
| **Activation** | Manual `poetry shell` | Automatic via `uv run` |
| **Command** | `poetry run python` | `uv run python` |

**Summary:** The differences are in **how dependencies are managed** (selective vs. all-at-once), not in how running processes handle in-memory packages. This is actually a **feature, not a bug** - it prevents running processes from suddenly breaking when dependencies change!

### Q: Can I install multiple extras simultaneously?

**A:** Yes, but with limitations:

```bash
# This works - installs both PyTorch 2.1 AND sklearn
uv sync --extra pytorch_2_1 --extra sklearn

# This fails - conflicts with pytorch_2_0 vs pytorch_2_1
uv sync --extra pytorch_2_0 --extra pytorch_2_1
```

The `pyproject.toml` has conflict rules that prevent incompatible combinations.

### Q: How is uv different from Poetry?

**A:** Key differences:

| Aspect | Poetry | uv |
|--------|--------|-----|
| Virtual Environment | Single `.venv` | Single `.venv` |
| Dependency Management | All deps installed | Selective installation |
| Activation | Manual activation required | Automatic via `uv run` |
| Extras | All extras available | Only specified extras |
| Lock File | `poetry.lock` | `uv.lock` |
| Commands | `poetry install`, `poetry run` | `uv sync`, `uv run` |

## Worker Management

### Q: Can I run multiple workers with different versions simultaneously?

**A:** Yes! You can run multiple workers with different frameworks:

```bash
# Terminal 1: PyTorch 2.0 CPU
export WORKER_TYPE=pytorch-2.0
export MODEL_FRAMEWORK=pytorch
uv sync --extra pytorch_2_0
uv run python src/worker.py

# Terminal 2: PyTorch 2.1 GPU
export WORKER_TYPE=pytorch-2.1-gpu
export MODEL_FRAMEWORK=pytorch
export USE_GPU=true
uv sync --extra pytorch_2_1_gpu
uv run python src/worker.py

# Terminal 3: TensorFlow
export WORKER_TYPE=tensorflow
export MODEL_FRAMEWORK=tensorflow
uv sync --extra tensorflow
uv run python src/worker.py
```

### Q: How do workers know which jobs to process?

**A:** Each worker listens to specific Redis queues based on its `WORKER_TYPE`:

| Worker Type | Training Queue | Inference Queue |
|-------------|----------------|-----------------|
| `pytorch-2.0` | `ai:training:pytorch-2.0:queue` | `ai:inference:pytorch-2.0:queue` |
| `pytorch-2.1-gpu` | `ai:training:pytorch-2.1-gpu:queue` | `ai:inference:pytorch-2.1-gpu:queue` |
| `tensorflow` | `ai:training:tensorflow:queue` | `ai:inference:tensorflow:queue` |

### Q: What environment variables do workers need?

**A:** Required variables:
- `WORKER_TYPE`: The type of worker (e.g., `pytorch-2.1-gpu`)
- `MODEL_FRAMEWORK`: The ML framework (e.g., `pytorch`, `tensorflow`)

Optional variables:
- `USE_GPU`: Set to `true` for GPU workers
- `REDIS_HOST`: Redis server host (default: `localhost`)
- `MAX_CONCURRENT_JOBS`: Maximum concurrent jobs (default: `2`)

### Q: How do I scale workers?

**A:** Multiple approaches:

**Docker Compose scaling:**
```bash
docker compose up -d --scale ai-worker-pytorch-2.1=3
```

**Manual scaling:**
```bash
# Run multiple instances of the same worker type
export WORKER_TYPE=pytorch-2.1
uv run python src/worker.py  # Terminal 1
uv run python src/worker.py  # Terminal 2
uv run python src/worker.py  # Terminal 3
```

## API Usage

### Q: How do I submit a training job?

**A:** Use the `/jobs/training` endpoint:

```bash
curl -X POST "http://localhost:8000/jobs/training" \
     -H "Content-Type: application/json" \
     -d '{
       "model_type": "bert",
       "data_path": "/data/sentiment_analysis.csv",
       "hyperparameters": {
         "epochs": 5,
         "learning_rate": 0.001,
         "batch_size": 16
       },
       "description": "BERT sentiment analysis training",
       "requires_gpu": false
     }'
```

### Q: How do I check job status?

**A:** Use the `/jobs/{job_id}/status` endpoint:

```bash
curl "http://localhost:8000/jobs/job_12345678-1234-1234-1234-123456789012/status"
```

### Q: How does the backend route jobs to workers?

**A:** The backend automatically routes jobs based on:

1. **Framework Override**: If `framework_override` is specified
2. **Model Type + GPU**: Route based on model type and GPU requirements
3. **Default**: Fall back to PyTorch 2.1 CPU worker

**Routing Examples:**
- `bert` + `requires_gpu: false` → PyTorch 2.1 CPU worker
- `resnet` + `requires_gpu: true` → PyTorch 2.0 GPU worker
- `inception` → TensorFlow worker
- `random_forest` → Scikit-learn worker

### Q: What are the supported model types?

**A:** The system supports various model types:

**PyTorch Models:**
- `bert`, `transformer`, `gpt` (NLP)
- `resnet`, `vgg`, `alexnet` (Computer Vision)

**TensorFlow Models:**
- `inception`, `mobilenet`, `efficientnet` (Computer Vision)

**Scikit-learn Models:**
- `random_forest`, `svm`, `logistic_regression` (Traditional ML)

## Troubleshooting

### Q: Worker exits immediately after loading PyTorch

**A:** This usually means the `main` function or `run` method is missing from the worker. The worker loads dependencies but doesn't start the job processing loop.

**Solution:** Make sure the worker.py file has the complete `main` function and `run` method.

### Q: NumPy compatibility errors

**A:** This happens when packages were compiled with NumPy 1.x but you're running NumPy 2.x.

**Solution:** Pin NumPy to version 1.x in `pyproject.toml`:
```toml
dependencies = [
    "numpy>=1.24.0,<2.0.0",
    # ... other deps
]
```

### Q: Redis connection errors

**A:** Make sure Redis is running:

```bash
# Check if Redis is running
redis-cli ping

# Start Redis
# macOS: brew services start redis
# Ubuntu: sudo systemctl start redis-server
```

### Q: GPU workers not using GPU

**A:** Check several things:

```bash
# Check CUDA availability
uv run python -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
nvidia-smi

# Verify environment variables
echo $USE_GPU
echo $WORKER_TYPE
```

### Q: Jobs not being processed

**A:** Check the worker and queue status:

```bash
# Check if worker is running
ps aux | grep "python src/worker.py"

# Check queue lengths
redis-cli LLEN ai:training:pytorch-2.1:queue

# Check worker logs
tail -f worker.log
```

### Q: Import errors after switching extras

**A:** This happens when you switch extras but don't restart running processes.

**Solution:** Stop and restart the worker:
```bash
# Stop current worker (Ctrl+C)
# Switch dependencies
uv sync --extra pytorch_2_1
# Restart worker
uv run python src/worker.py
```

## Deployment

### Q: How do I deploy this system?

**A:** Multiple deployment options:

**Docker Compose (Recommended):**
```bash
./scripts/docker-compose.sh up -d
```

**Local Development:**
```bash
./scripts/run_local.sh
```

**Production:**
- Use Docker Compose with proper environment configuration
- Set up monitoring and logging
- Configure reverse proxy (nginx)
- Use managed Redis service

### Q: How do I monitor the system?

**A:** Use the health check endpoints:

```bash
# API health
curl http://localhost:8000/health

# Check worker status
docker ps  # For Docker deployment
ps aux | grep worker  # For local deployment
```

### Q: How do I scale for production?

**A:** Production scaling strategies:

**Horizontal Scaling:**
```bash
# Scale specific worker types
docker compose up -d --scale ai-worker-pytorch-2.1=5
docker compose up -d --scale ai-worker-pytorch-2.1-gpu=3
```

**Load Balancing:**
- Use multiple backend instances behind a load balancer
- Distribute workers across multiple machines
- Use managed Redis cluster for high availability

**Resource Management:**
- Set appropriate memory and CPU limits
- Monitor resource usage
- Implement auto-scaling based on queue length

### Q: How do I handle GPU memory management?

**A:** The system includes automatic GPU memory management:

- **Memory Allocation**: Workers allocate GPU memory before training
- **Memory Cleanup**: Memory is freed after job completion
- **Queue Management**: Jobs wait for available GPU memory
- **Parameter Optimization**: Batch sizes are optimized based on available memory

See [GPU Memory Management](GPU_MEMORY_MANAGEMENT.md) for detailed information.

## Additional Resources

- [API Documentation](API.md) - Complete API reference
- [Worker Management](WORKER_MANAGEMENT.md) - Worker configuration guide
- [Architecture](ARCHITECTURE.md) - System architecture details
- [Deployment](DEPLOYMENT.md) - Deployment guides

## Getting Help

If you have questions not covered in this FAQ:

1. Check the [API Documentation](API.md)
2. Review the [Worker Management Guide](WORKER_MANAGEMENT.md)
3. Look at the example scripts in the `scripts/` directory
4. Check the Docker Compose configuration for deployment examples
