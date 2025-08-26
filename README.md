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
```bash
cd ai-worker
uv sync
uv run python src/worker.py
```

## Deployment Options

### 🐳 Docker Compose (Development/Staging)
```bash
# Automatic detection (recommended)
./scripts/docker-compose.sh up -d

# Manual commands:
# Modern Docker (Docker Desktop 4.0+)
docker compose up -d

# Legacy Docker
docker-compose up -d
```

### 🏠 Local Development (No Docker)
```bash
./scripts/run_local.sh
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
