# Dynamic Worker Management

This document describes the dynamic worker spawning system that allows workers to be created on-demand when jobs are submitted, rather than running all workers continuously.

## Overview

The dynamic worker system consists of three main components:

1. **Worker Manager Service**: Manages the lifecycle of worker containers
2. **Backend API**: Triggers worker spawning when jobs are submitted
3. **Worker Containers**: Created dynamically based on job requirements

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client        │    │   Backend API    │    │ Worker Manager  │
│                 │    │                  │    │                 │
│ Submit Job ────►│    │ Ensure Worker ──►│    │ Spawn Worker    │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Redis Queue    │    │ Docker Engine   │
                       │                  │    │                 │
                       │ Queue Job        │    │ Create Container│
                       └──────────────────┘    └─────────────────┘
```

## Benefits

### Resource Efficiency
- **No idle workers**: Workers are only created when needed
- **Reduced memory usage**: No continuous memory consumption from unused workers
- **Cost optimization**: Lower infrastructure costs in cloud environments

### Scalability
- **On-demand scaling**: Workers are created based on actual demand
- **Framework-specific**: Only the required framework workers are spawned
- **Auto-cleanup**: Idle workers are automatically terminated

### Flexibility
- **Dynamic framework selection**: Workers are spawned based on job requirements
- **GPU optimization**: GPU workers are only created when GPU is required
- **Easy management**: Centralized worker lifecycle management

## Components

### Worker Manager Service

**Location**: `worker-manager/src/worker_manager.py`

**Responsibilities**:
- Spawn worker containers on demand
- Monitor worker health and activity
- Clean up idle workers automatically
- Manage worker lifecycle (start, stop, scale)

**API Endpoints**:
- `POST /workers/start/{worker_type}` - Start a specific worker type
- `POST /workers/stop/{worker_id}` - Stop a specific worker
- `GET /workers/status` - Get status of all workers
- `POST /workers/ensure/{worker_type}` - Ensure worker availability

### Backend API Integration

**Modified**: `backend/src/main.py`

**Changes**:
- Added `ensure_worker_available()` function
- Modified job submission to trigger worker spawning
- Added worker manager communication via HTTP

### Worker Configurations

**Supported Worker Types**:
```python
WORKER_CONFIGS = {
    "pytorch-2.0": {
        "dockerfile": "Dockerfile.pytorch-2.0",
        "framework": "pytorch",
        "version": "2.0.0",
        "gpu": False,
        "startup_time": 30,
        "idle_timeout": 300,  # 5 minutes
    },
    "pytorch-2.1": {
        "dockerfile": "Dockerfile.pytorch-2.1",
        "framework": "pytorch",
        "version": "2.1.0",
        "gpu": False,
        "startup_time": 30,
        "idle_timeout": 300,
    },
    "pytorch-2.0-gpu": {
        "dockerfile": "Dockerfile.pytorch-2.0-gpu",
        "framework": "pytorch",
        "version": "2.0.0",
        "gpu": True,
        "startup_time": 45,
        "idle_timeout": 600,  # 10 minutes
    },
    "pytorch-2.1-gpu": {
        "dockerfile": "Dockerfile.pytorch-2.1-gpu",
        "framework": "pytorch",
        "version": "2.1.0",
        "gpu": True,
        "startup_time": 45,
        "idle_timeout": 600,
    },
    "tensorflow": {
        "dockerfile": "Dockerfile.tensorflow",
        "framework": "tensorflow",
        "version": "2.13.0",
        "gpu": False,
        "startup_time": 40,
        "idle_timeout": 300,
    },
    "sklearn": {
        "dockerfile": "Dockerfile.sklearn",
        "framework": "sklearn",
        "version": "1.3.0",
        "gpu": False,
        "startup_time": 20,
        "idle_timeout": 300,
    },
}
```

## Usage

### Starting the System

```bash
# Start with dynamic workers
./scripts/run_dynamic.sh

# Or manually with docker-compose
docker-compose -f docker-compose.dynamic.yml up -d
```

### Submitting Jobs

When you submit a job, the system will:

1. **Determine worker type** based on model type and requirements
2. **Check for available workers** of that type
3. **Spawn a new worker** if none are available
4. **Queue the job** for processing
5. **Clean up idle workers** after timeout

```bash
# Submit a training job (will spawn worker automatically)
curl -X POST http://localhost:8000/jobs/training \
  -H 'Content-Type: application/json' \
  -d '{
    "model_type": "bert",
    "data_path": "/data/train.csv",
    "requires_gpu": false
  }'
```

### Monitoring Workers

```bash
# Check worker status
curl http://localhost:8001/workers/status

# Response example:
{
  "total_workers": 2,
  "active_workers": 2,
  "workers_by_type": {
    "pytorch-2.1": 1,
    "tensorflow": 1
  },
  "worker_details": {
    "pytorch-2.1-1703123456": {
      "worker_id": "pytorch-2.1-1703123456",
      "worker_type": "pytorch-2.1",
      "container_name": "ai-worker-pytorch-2.1-1703123456",
      "status": "running",
      "created_at": "2024-01-15T10:30:00Z",
      "last_activity": "2024-01-15T10:35:00Z",
      "job_count": 1,
      "max_jobs": 2
    }
  }
}
```

## Configuration

### Environment Variables

**Worker Manager**:
- `REDIS_HOST`: Redis server host (default: `redis`)
- `REDIS_PORT`: Redis server port (default: `6379`)
- `REDIS_DB`: Redis database number (default: `0`)

**Backend API**:
- `WORKER_MANAGER_URL`: Worker manager service URL (default: `http://worker-manager:8001`)

### Worker Timeouts

- **CPU Workers**: 5 minutes idle timeout
- **GPU Workers**: 10 minutes idle timeout (longer due to startup cost)
- **Startup Time**: 20-45 seconds depending on framework

### Scaling Limits

- **Maximum workers per type**: 3
- **Maximum concurrent jobs per worker**: 2
- **Auto-cleanup interval**: 60 seconds

## Comparison: Static vs Dynamic

| Aspect | Static Workers | Dynamic Workers |
|--------|----------------|-----------------|
| **Resource Usage** | High (always running) | Low (on-demand) |
| **Startup Time** | Immediate | 20-45 seconds |
| **Cost** | Higher | Lower |
| **Complexity** | Simple | Moderate |
| **Scalability** | Manual | Automatic |
| **GPU Efficiency** | Poor (idle GPU usage) | Good (GPU only when needed) |

## Migration from Static to Dynamic

### Step 1: Update docker-compose

```bash
# Stop static workers
docker-compose down

# Start dynamic system
docker-compose -f docker-compose.dynamic.yml up -d
```

### Step 2: Update client code

No changes needed! The API remains the same.

### Step 3: Monitor and adjust

- Monitor worker spawning patterns
- Adjust timeouts based on usage
- Scale limits if needed

## Troubleshooting

### Common Issues

**Workers not spawning**:
```bash
# Check worker manager logs
docker-compose -f docker-compose.dynamic.yml logs worker-manager

# Check Docker socket permissions
ls -la /var/run/docker.sock
```

**Jobs stuck in queue**:
```bash
# Check Redis queue lengths
redis-cli
> LLEN ai:training:pytorch-2.1:queue

# Check worker status
curl http://localhost:8001/workers/status
```

**High startup latency**:
- Consider pre-warming workers for common job types
- Optimize Docker images for faster startup
- Use faster storage (SSD) for container images

### Performance Tuning

**Reduce startup time**:
- Use multi-stage Docker builds
- Pre-install common dependencies
- Use smaller base images

**Optimize cleanup**:
- Adjust idle timeouts based on usage patterns
- Monitor worker utilization
- Set appropriate scaling limits

## Future Enhancements

### Planned Features

1. **Predictive Scaling**: Pre-spawn workers based on historical patterns
2. **Worker Pooling**: Maintain minimum worker pools for common types
3. **Resource Monitoring**: CPU/GPU usage-based scaling decisions
4. **Cost Optimization**: Automatic framework selection based on cost
5. **Health Checks**: Advanced worker health monitoring and recovery

### Integration Opportunities

1. **Kubernetes**: Native pod management instead of Docker
2. **Cloud Providers**: Integration with AWS ECS, GCP Cloud Run
3. **Monitoring**: Prometheus/Grafana integration
4. **Alerting**: Slack/email notifications for worker issues

## Best Practices

### Development

1. **Test locally**: Use dynamic workers in development
2. **Monitor patterns**: Understand your job submission patterns
3. **Optimize images**: Keep Docker images small and fast
4. **Set appropriate timeouts**: Balance resource usage vs startup time

### Production

1. **Monitor costs**: Track resource usage and costs
2. **Set scaling limits**: Prevent runaway worker creation
3. **Health monitoring**: Set up alerts for worker failures
4. **Backup strategy**: Ensure worker state persistence
5. **Security**: Secure Docker socket access

## Conclusion

The dynamic worker system provides significant benefits in terms of resource efficiency and cost optimization while maintaining the same API interface. It's particularly beneficial for:

- **Variable workloads**: Jobs submitted irregularly
- **Cost-sensitive environments**: Cloud deployments with usage-based pricing
- **GPU resources**: Expensive GPU instances that should only run when needed
- **Multi-framework usage**: Different frameworks used at different times

The system automatically handles worker lifecycle management, making it transparent to users while providing better resource utilization.
