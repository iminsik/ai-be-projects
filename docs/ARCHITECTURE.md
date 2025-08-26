# Architecture Documentation

## System Overview

This project implements a microservices-based AI job queue system that separates concerns between the FastAPI backend and AI training/inference workers. This design ensures independent dependency management, scalability, and maintainability.

## Architecture Components

### 1. FastAPI Backend Service (`backend/`)

**Purpose**: HTTP API server that handles job submission and status queries.

**Key Features**:
- RESTful API endpoints for job management
- Redis integration for job queue and status storage
- Async/await support for high concurrency
- Pydantic models for request/response validation

**Dependencies** (managed by uv):
- `fastapi>=0.104.0` - Web framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `redis>=5.0.0` - Redis client
- `pydantic>=2.5.0` - Data validation

**API Endpoints**:
- `POST /jobs/training` - Submit training job
- `POST /jobs/inference` - Submit inference job
- `GET /jobs/{job_id}/status` - Get job status
- `GET /jobs` - List all jobs
- `GET /health` - Health check

### 2. AI Worker Service (`ai-worker/`)

**Purpose**: Processes training and inference jobs from Redis queues.

**Key Features**:
- Concurrent processing of training and inference queues
- Model caching and storage management
- Progress tracking and status updates
- Graceful shutdown handling

**Dependencies** (managed by uv):
- `redis>=5.0.0` - Redis client
- `torch>=2.0.0` - PyTorch for ML
- `transformers>=4.30.0` - Hugging Face transformers
- `scikit-learn>=1.3.0` - Traditional ML
- `numpy>=1.24.0` - Numerical computing
- `pandas>=2.0.0` - Data manipulation

**Optional Dependencies**:
- `gpu` extra: CUDA-enabled PyTorch for GPU acceleration
- `dev` extra: Development and testing tools

### 3. Redis Service

**Purpose**: Message broker and job status storage.

**Key Features**:
- Job queue management (training and inference queues)
- Job status persistence with TTL
- Pub/sub capabilities for future extensions
- Data persistence with AOF (Append-Only File)

## Data Flow

```
1. Client submits job via FastAPI
   ↓
2. FastAPI creates job status in Redis
   ↓
3. FastAPI pushes job to appropriate queue
   ↓
4. AI Worker picks up job from queue
   ↓
5. AI Worker updates status to "running"
   ↓
6. AI Worker processes job (training/inference)
   ↓
7. AI Worker updates status with results
   ↓
8. Client can query job status via FastAPI
```

## Redis Data Structure

### Queues
- `ai:training:queue` - List of pending training jobs
- `ai:inference:queue` - List of pending inference jobs

### Job Status
- `ai:job:status:{job_id}` - Hash containing job status and metadata
  - `status`: pending, running, completed, failed
  - `job_type`: training, inference
  - `created_at`: ISO timestamp
  - `updated_at`: ISO timestamp
  - `result`: Job results (if completed)
  - `error`: Error message (if failed)
  - `metadata`: Job-specific metadata

## Benefits of This Architecture

### 1. Dependency Isolation

**Problem**: AI libraries often have complex dependency requirements that conflict with web framework dependencies.

**Solution**: Separate services with independent dependency management:
- Backend uses minimal dependencies focused on web serving
- AI worker can use any Python version and ML libraries
- Each service can be updated independently

### 2. Scalability

**Horizontal Scaling**: Multiple AI workers can be deployed:
```bash
# Scale AI workers independently
docker-compose up --scale ai-worker=5
```

**Resource Optimization**: Different worker types for different workloads:
- CPU workers for lightweight inference
- GPU workers for heavy training
- Memory-optimized workers for large models

### 3. Technology Flexibility

**AI Worker Flexibility**:
- Can use different ML frameworks (PyTorch, TensorFlow, etc.)
- Can run on different Python versions
- Can use different hardware (CPU, GPU, TPU)

**Backend Flexibility**:
- Can be replaced with any web framework
- Can integrate with different databases
- Can add authentication, rate limiting, etc.

### 4. Development Workflow

**Independent Development**:
```bash
# Develop backend only
cd backend && uv run python -m uvicorn src.main:app --reload

# Develop AI worker only
cd ai-worker && uv run python src/worker.py

# Test integration
docker-compose up -d
```

**Testing**:
- Unit tests for each service independently
- Integration tests with Redis
- End-to-end tests with full stack

## Deployment Options

### 1. Docker Compose (Development/Staging)
```bash
docker-compose up -d
```

### 2. Kubernetes (Production)
```yaml
# Separate deployments for each service
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-backend
spec:
  replicas: 3
  # ... backend configuration

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-worker
spec:
  replicas: 5
  # ... worker configuration
```

### 3. Serverless (Future)
- Backend: AWS Lambda + API Gateway
- AI Worker: AWS Batch or Google Cloud Run
- Queue: AWS SQS or Google Cloud Tasks

## Monitoring and Observability

### Health Checks
- Backend: `/health` endpoint
- AI Worker: Redis connection status
- Redis: Built-in health check

### Logging
- Structured logging with correlation IDs
- Centralized log aggregation
- Job-specific log streams

### Metrics
- Job queue length
- Processing time
- Success/failure rates
- Resource utilization

## Security Considerations

### 1. Network Security
- Services communicate via internal Docker network
- Redis not exposed to external traffic
- API gateway for external access

### 2. Data Security
- Model files stored in secure volumes
- Job data encrypted in transit
- Access control for model storage

### 3. Authentication
- API authentication (JWT, OAuth)
- Worker authentication with Redis
- Model access control

## Future Enhancements

### 1. Advanced Queue Features
- Priority queues
- Job scheduling
- Dead letter queues
- Retry mechanisms

### 2. Model Management
- Model versioning
- Model registry
- A/B testing support
- Model performance monitoring

### 3. Distributed Training
- Multi-GPU training
- Multi-node training
- Federated learning support

### 4. Real-time Features
- WebSocket support for real-time updates
- Streaming inference
- Live model updates
