# AI Training & Inference Job Queue System

A microservices-based system that separates FastAPI backend from AI training/inference workers, allowing independent dependency management and scaling.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Redis         │    │   AI Worker     │
│   Backend       │◄──►│   (Job Queue)   │◄──►│   Service       │
│                 │    │                 │    │                 │
│ - Job submission│    │ - Job storage   │    │ - Training      │
│ - Status check  │    │ - Result cache  │    │ - Inference     │
│ - API endpoints │    │ - Pub/Sub       │    │ - Independent   │
└─────────────────┘    └─────────────────┘    │   dependencies  │
                                              └─────────────────┘
```

## Key Features

- **Independent Dependencies**: Backend and AI workers use separate dependency management
- **Multiple ML Frameworks**: Support for PyTorch 2.0/2.1, TensorFlow, and Scikit-learn
- **GPU Support**: CUDA-enabled PyTorch workers for GPU acceleration
- **Scalable**: Multiple AI workers can be deployed independently
- **Job Queue**: Redis-based job queue with status tracking
- **Docker Support**: Easy deployment with Docker Compose
- **Modern Python**: Uses uv for fast dependency management

## Project Structure

```
.
├── backend/                 # FastAPI service
│   ├── pyproject.toml      # Backend dependencies (uv)
│   ├── src/
│   │   └── main.py         # FastAPI application
│   └── Dockerfile
├── ai-worker/              # AI training/inference service
│   ├── pyproject.toml      # AI dependencies (uv)
│   ├── src/
│   │   └── worker.py       # AI worker implementation
│   └── Dockerfile
├── docker-compose.yml      # Service orchestration
└── README.md
```

## Quick Start

### Option 1: Docker Compose (Recommended for Development)

1. **Start all services:**
   ```bash
   docker-compose up -d
   ```

2. **Submit a training job:**
   ```bash
   curl -X POST "http://localhost:8000/jobs/training" \
        -H "Content-Type: application/json" \
        -d '{"model_type": "transformer", "data_path": "/data/train.csv"}'
   ```

3. **Submit an inference job:**
   ```bash
   curl -X POST "http://localhost:8000/jobs/inference" \
        -H "Content-Type: application/json" \
        -d '{"model_id": "model_123", "input_data": "sample text"}'
   ```

4. **Check job status:**
   ```bash
   curl "http://localhost:8000/jobs/{job_id}/status"
   ```

### Option 2: Local Development (No Docker)

1. **Install prerequisites:**
   ```bash
   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install Redis
   # macOS: brew install redis && brew services start redis
   # Ubuntu: sudo apt install redis-server && sudo systemctl start redis-server
   ```

2. **Run the system:**
   ```bash
   ./scripts/run_local.sh
   ```

3. **Test the API:**
   ```bash
   python scripts/test_api.py
   ```

## Development

### Backend Development
```bash
cd backend
uv sync
uv run python -m uvicorn src.main:app --reload
```

### AI Worker Development

**For PyTorch 2.0:**
```bash
cd ai-worker
uv sync --extra pytorch_2_0
uv run python src/worker.py
```

**For PyTorch 2.1:**
```bash
cd ai-worker
uv sync --extra pytorch_2_1
uv run python src/worker.py
```

**For TensorFlow:**
```bash
cd ai-worker
uv sync --extra tensorflow
uv run python src/worker.py
```

**For Scikit-learn:**
```bash
cd ai-worker
uv sync --extra sklearn
uv run python src/worker.py
```

**For GPU support (PyTorch 2.0):**
```bash
cd ai-worker
uv sync --extra pytorch_2_0_gpu
uv run python src/worker.py
```

**For GPU support (PyTorch 2.1):**
```bash
cd ai-worker
uv sync --extra pytorch_2_1_gpu
uv run python src/worker.py
```

## Deployment Options

### 🐳 Docker Compose (Development/Staging)

**Start all services:**
```bash
# Automatic detection (recommended)
./scripts/docker-compose.sh up -d

# Manual commands:
# Modern Docker (Docker Desktop 4.0+)
docker compose up -d

# Legacy Docker
docker-compose up -d
```

**Start specific AI worker types:**
```bash
# PyTorch 2.0 workers only
docker compose up -d redis backend ai-worker-pytorch-2.0

# PyTorch 2.1 workers only  
docker compose up -d redis backend ai-worker-pytorch-2.1

# TensorFlow workers only
docker compose up -d redis backend ai-worker-tensorflow

# Scikit-learn workers only
docker compose up -d redis backend ai-worker-sklearn

# GPU workers (PyTorch 2.0)
docker compose up -d redis backend ai-worker-pytorch-2.0-gpu

# GPU workers (PyTorch 2.1)
docker compose up -d redis backend ai-worker-pytorch-2.1-gpu
```

### 🏠 Local Development (No Docker)

**Start the system:**
```bash
./scripts/run_local.sh
```

**Manual setup:**
```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start Backend
cd backend
uv sync
uv run python -m uvicorn src.main:app --reload

# Terminal 3: Start AI Worker (choose one)
cd ai-worker

# For PyTorch 2.0
uv sync --extra pytorch_2_0
uv run python src/worker.py

# For PyTorch 2.1  
uv sync --extra pytorch_2_1
uv run python src/worker.py

# For TensorFlow
uv sync --extra tensorflow
uv run python src/worker.py

# For Scikit-learn
uv sync --extra sklearn
uv run python src/worker.py
```

### 🖥️ Traditional Server
```bash
# Install system dependencies
sudo apt install python3.11 redis-server nginx

# Setup services
./scripts/setup_dev.sh
sudo systemctl start redis-server
sudo systemctl start ai-backend
sudo systemctl start ai-worker
```

### ☁️ Cloud Deployment
- **AWS**: EC2 with managed Redis
- **GCP**: Compute Engine with Cloud Memorystore
- **Azure**: VM with Azure Cache for Redis
- **Kubernetes**: Native K8s deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed deployment guides.

## API Endpoints

- `POST /jobs/training` - Submit training job
- `POST /jobs/inference` - Submit inference job
- `GET /jobs/{job_id}/status` - Get job status
- `GET /jobs` - List all jobs

## Benefits of This Architecture

1. **Dependency Isolation**: Backend and AI components can use different Python versions and packages
2. **Independent Scaling**: Scale AI workers independently based on workload
3. **Technology Flexibility**: Each service can use different frameworks/libraries
4. **Easy Deployment**: Docker Compose handles service orchestration
5. **Development Flexibility**: Develop and test services independently
