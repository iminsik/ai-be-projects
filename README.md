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
