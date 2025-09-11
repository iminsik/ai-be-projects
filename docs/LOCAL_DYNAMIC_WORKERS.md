# Local Dynamic Worker Management

This document describes the local dynamic worker spawning system that allows workers to be created on-demand as local processes, without requiring Docker.

## Overview

The local dynamic worker system is perfect for development and testing environments where you want the benefits of dynamic worker spawning without the overhead of Docker containers.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client        │    │   Backend API    │    │ Local Worker    │
│                 │    │                  │    │ Manager         │
│ Submit Job ────►│    │ Ensure Worker ──►│    │ Spawn Process   │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Redis Queue    │    │ Local Process   │
                       │                  │    │                 │
                       │ Queue Job        │    │ Start Worker    │
                       └──────────────────┘    └─────────────────┘
```

## Benefits

### Development Friendly
- **No Docker required**: Works with just Python and Redis
- **Fast startup**: 3-10 seconds vs 20-45 seconds for Docker
- **Easy debugging**: Direct access to worker processes
- **Resource efficient**: Lower memory usage than containers

### Framework Support
- **PyTorch 2.0 & 2.1**: Both CPU and GPU variants
- **TensorFlow 2.13**: Full TensorFlow ecosystem
- **Scikit-learn 1.3**: Traditional ML algorithms
- **GPU Support**: CUDA support when available

## Components

### Local Worker Manager

**Location**: `worker-manager/src/local_worker_manager.py`

**Responsibilities**:
- Spawn local worker processes on demand
- Monitor worker health and activity
- Clean up idle workers automatically
- Manage worker lifecycle (start, stop, scale)

**API Endpoints**:
- `POST /workers/start/{worker_type}` - Start a specific worker type
- `POST /workers/stop/{worker_id}` - Stop a specific worker
- `GET /workers/status` - Get status of all workers
- `POST /workers/ensure/{worker_type}` - Ensure worker availability

### Worker Process Management

**Process Spawning**:
- Uses `subprocess.Popen` to start worker processes
- Sets appropriate environment variables
- Monitors process health via PID tracking
- Handles process cleanup on shutdown

**Environment Variables**:
```bash
WORKER_TYPE=pytorch-2.1
MODEL_FRAMEWORK=pytorch
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
USE_GPU=true  # For GPU workers
```

## Usage

### Starting the System

```bash
# Start with local dynamic workers
./scripts/run_local_dynamic.sh
```

This will:
- Start Redis, Backend API, and Local Worker Manager
- Workers will be spawned as local processes when needed
- Idle workers are cleaned up after 5-10 minutes

### Submitting Jobs

When you submit a job, the system will:

1. **Determine worker type** based on model type and requirements
2. **Check for available workers** of that type
3. **Spawn a new worker process** if none are available
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
      "process_id": 12345,
      "status": "running",
      "created_at": "2024-01-15T10:30:00Z",
      "last_activity": "2024-01-15T10:35:00Z",
      "job_count": 1,
      "max_jobs": 2,
      "working_directory": "/path/to/ai-worker"
    }
  }
}
```

## Configuration

### Worker Types

```python
LOCAL_WORKER_CONFIGS = {
    "pytorch-2.0": {
        "framework": "pytorch",
        "version": "2.0.0",
        "gpu": False,
        "startup_time": 5,
        "idle_timeout": 300,
        "extra": "pytorch_2_0",
    },
    "pytorch-2.1": {
        "framework": "pytorch",
        "version": "2.1.0",
        "gpu": False,
        "startup_time": 5,
        "idle_timeout": 300,
    },
    "pytorch-2.0-gpu": {
        "framework": "pytorch",
        "version": "2.0.0",
        "gpu": True,
        "startup_time": 8,
        "idle_timeout": 600,
        "extra": "pytorch_2_0_gpu",
    },
    "pytorch-2.1-gpu": {
        "framework": "pytorch",
        "version": "2.1.0",
        "gpu": True,
        "startup_time": 8,
        "idle_timeout": 600,
    },
    "tensorflow": {
        "framework": "tensorflow",
        "version": "2.13.0",
        "gpu": False,
        "startup_time": 10,
        "idle_timeout": 300,
        "extra": "tensorflow",
    },
    "sklearn": {
        "framework": "sklearn",
        "version": "1.3.0",
        "gpu": False,
        "startup_time": 3,
        "idle_timeout": 300,
        "extra": "sklearn",
    },
}
```

### Environment Variables

**Local Worker Manager**:
- `REDIS_HOST`: Redis server host (default: `localhost`)
- `REDIS_PORT`: Redis server port (default: `6379`)
- `REDIS_DB`: Redis database number (default: `0`)

**Backend API**:
- `WORKER_MANAGER_URL`: Worker manager service URL (default: `http://localhost:8001`)
- `WORKER_MANAGER_TYPE`: Set to `local` for local workers

## Comparison: Docker vs Local Dynamic Workers

| Aspect | Docker Dynamic | Local Dynamic |
|--------|----------------|---------------|
| **Startup Time** | 20-45 seconds | 3-10 seconds |
| **Resource Usage** | Higher (containers) | Lower (processes) |
| **Isolation** | Complete | Process-level |
| **Dependencies** | Docker required | Python + Redis only |
| **Debugging** | Container logs | Direct process access |
| **Best For** | Production | Development/Testing |
| **GPU Support** | Full | Limited (depends on host) |

## Testing

### Automated Testing

```bash
# Run the test suite
python scripts/test_local_dynamic.py
```

This will test:
- Service health checks
- Worker spawning
- Job submission
- Worker status monitoring
- Cleanup

### Manual Testing

```bash
# Start the system
./scripts/run_local_dynamic.sh

# In another terminal, test workers
curl -X POST http://localhost:8001/workers/start/pytorch-2.1
curl http://localhost:8001/workers/status
curl -X POST http://localhost:8001/workers/stop/{worker_id}
```

## Troubleshooting

### Common Issues

**Workers not starting**:
```bash
# Check if uv is installed
uv --version

# Check if Redis is running
redis-cli ping

# Check worker manager logs
ps aux | grep local_worker_manager
```

**Import errors**:
```bash
# Make sure dependencies are installed
cd ai-worker
uv sync --extra pytorch_2_1
```

**Process cleanup**:
```bash
# Kill all worker processes
pkill -f "run_worker.py"

# Check for zombie processes
ps aux | grep python
```

### Debugging

**Enable debug logging**:
```bash
export LOG_LEVEL=DEBUG
./scripts/run_local_dynamic.sh
```

**Monitor worker processes**:
```bash
# Watch worker processes
watch -n 1 'ps aux | grep run_worker'

# Monitor Redis queues
redis-cli monitor
```

## Performance Tuning

### Startup Time Optimization

1. **Pre-install dependencies**: Ensure all framework dependencies are installed
2. **Use faster storage**: SSD for better I/O performance
3. **Optimize imports**: Lazy load heavy libraries

### Memory Management

1. **Set appropriate limits**: Configure `max_jobs` per worker
2. **Monitor memory usage**: Use `htop` or `ps` to monitor
3. **Cleanup idle workers**: Adjust `idle_timeout` based on usage

### Scaling

1. **Adjust worker limits**: Change max workers per type
2. **Optimize timeouts**: Balance resource usage vs responsiveness
3. **Monitor patterns**: Understand your job submission patterns

## Best Practices

### Development

1. **Use local workers**: Perfect for development and testing
2. **Monitor resource usage**: Keep an eye on memory and CPU
3. **Test cleanup**: Ensure workers are properly cleaned up
4. **Use appropriate timeouts**: Don't set timeouts too low

### Production Considerations

1. **Use Docker workers**: Better isolation and resource management
2. **Monitor processes**: Set up process monitoring
3. **Handle failures**: Implement proper error handling
4. **Resource limits**: Set appropriate limits to prevent resource exhaustion

## Future Enhancements

### Planned Features

1. **Process monitoring**: Better health checks and monitoring
2. **Resource limits**: CPU and memory limits per worker
3. **Process pools**: Pre-warm worker processes
4. **Better cleanup**: More robust process cleanup
5. **Metrics**: Prometheus metrics for monitoring

### Integration Opportunities

1. **Process managers**: Integration with systemd, supervisor
2. **Container runtimes**: Support for podman, containerd
3. **Cloud providers**: Integration with cloud process services
4. **Monitoring**: Better integration with monitoring tools

## Conclusion

The local dynamic worker system provides an excellent development and testing environment for the AI Job Queue System. It offers:

- **Fast startup times** for quick iteration
- **No Docker dependency** for simpler setup
- **Easy debugging** with direct process access
- **Resource efficiency** for development machines
- **Full framework support** for all ML frameworks

It's perfect for development, testing, and small-scale deployments where Docker overhead is not desired.
