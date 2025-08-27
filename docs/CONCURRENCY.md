# Concurrency and Resource Management Guide

## Overview

The AI worker system is designed to handle multiple concurrent training and inference jobs based on available CPU and GPU resources. This guide explains how to configure and optimize the system for different hardware configurations.

## 🚀 **Concurrency Features**

### 1. **Automatic Resource Management**
- **Semaphore-based limiting**: Controls maximum concurrent jobs
- **Queue-based processing**: Jobs wait in Redis queues
- **Resource-aware scheduling**: Different limits for CPU vs GPU jobs

### 2. **Configurable Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_CONCURRENT_JOBS` | 2 | Maximum jobs per worker |
| `USE_GPU` | false | Enable GPU acceleration |
| `GPU_DEVICE` | 0 | GPU device ID |
| `JOB_TIMEOUT` | 3600 | Job timeout in seconds |

## 💻 **CPU-Only Configuration**

### Single CPU Worker
```bash
# Environment variables
MAX_CONCURRENT_JOBS=2
USE_GPU=false
JOB_TIMEOUT=3600

# Docker Compose
docker compose up -d ai-worker
```

### Multiple CPU Workers
```bash
# Scale CPU workers
docker compose up -d --scale ai-worker=4

# Or run multiple instances manually
MAX_CONCURRENT_JOBS=2 WORKER_ID=worker-1 docker compose up -d ai-worker
MAX_CONCURRENT_JOBS=2 WORKER_ID=worker-2 docker compose up -d ai-worker
```

### CPU Resource Guidelines

| CPU Cores | MAX_CONCURRENT_JOBS | Recommended Use |
|-----------|-------------------|-----------------|
| 2-4 cores | 1-2 | Light inference, small models |
| 4-8 cores | 2-4 | Medium training, multiple inference |
| 8+ cores | 4-8 | Heavy training, high-throughput inference |

## 🎮 **GPU Configuration**

### Single GPU Worker
```bash
# Environment variables
MAX_CONCURRENT_JOBS=1
USE_GPU=true
GPU_DEVICE=0

# Docker Compose (GPU)
docker compose up -d ai-worker-gpu
```

### Multiple GPU Workers
```bash
# Different GPUs
MAX_CONCURRENT_JOBS=1 USE_GPU=true GPU_DEVICE=0 WORKER_ID=gpu-worker-0 docker compose up -d ai-worker-gpu
MAX_CONCURRENT_JOBS=1 USE_GPU=true GPU_DEVICE=1 WORKER_ID=gpu-worker-1 docker compose up -d ai-worker-gpu
```

### GPU Resource Guidelines

| GPU Memory | MAX_CONCURRENT_JOBS | Model Types |
|------------|-------------------|-------------|
| 4-8 GB | 1 | Small models, inference only |
| 8-16 GB | 1-2 | Medium models, light training |
| 16+ GB | 2-4 | Large models, heavy training |

## 🔧 **Advanced Configuration**

### Mixed CPU/GPU Setup
```yaml
# docker-compose.yml
services:
  ai-worker-cpu:
    build: ./ai-worker
    environment:
      - MAX_CONCURRENT_JOBS=4
      - USE_GPU=false
    deploy:
      replicas: 2

  ai-worker-gpu:
    build: 
      context: ./ai-worker
      dockerfile: Dockerfile.gpu
    environment:
      - MAX_CONCURRENT_JOBS=2
      - USE_GPU=true
      - GPU_DEVICE=0
    runtime: nvidia
    deploy:
      replicas: 1
```

### Resource-Specific Job Queues
```python
# Custom queue routing based on job requirements
async def route_job(job_data):
    if job_data.get("requires_gpu", False):
        await redis_client.lpush("ai:gpu:queue", json.dumps(job_data))
    else:
        await redis_client.lpush("ai:cpu:queue", json.dumps(job_data))
```

## 📊 **Monitoring and Scaling**

### Worker Status Monitoring
```python
# Get worker status
worker_status = worker.get_status()
print(f"Active jobs: {worker_status['active_jobs']}")
print(f"Available slots: {worker_status['available_slots']}")
print(f"GPU enabled: {worker_status['gpu_enabled']}")
```

### Auto-scaling with Kubernetes
```yaml
# k8s/ai-worker-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-worker
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Load Balancing
```python
# Round-robin job distribution
async def distribute_jobs():
    workers = await get_available_workers()
    for job in pending_jobs:
        worker = workers[job_count % len(workers)]
        await send_job_to_worker(worker, job)
```

## 🎯 **Performance Optimization**

### 1. **Model Caching**
```python
# Keep frequently used models in memory
class ModelCache:
    def __init__(self, max_models=10):
        self.models = {}
        self.max_models = max_models
    
    async def get_model(self, model_id):
        if model_id not in self.models:
            if len(self.models) >= self.max_models:
                # Remove least recently used
                self.models.pop(next(iter(self.models)))
            self.models[model_id] = await load_model(model_id)
        return self.models[model_id]
```

### 2. **Batch Processing**
```python
# Process multiple inference jobs in batches
async def process_inference_batch(jobs, batch_size=32):
    batches = [jobs[i:i+batch_size] for i in range(0, len(jobs), batch_size)]
    results = []
    for batch in batches:
        batch_result = await model.predict_batch(batch)
        results.extend(batch_result)
    return results
```

### 3. **Resource Preemption**
```python
# Prioritize jobs based on resource requirements
class JobScheduler:
    def __init__(self):
        self.high_priority_queue = asyncio.Queue()
        self.normal_priority_queue = asyncio.Queue()
    
    async def schedule_job(self, job):
        if job.get("priority") == "high":
            await self.high_priority_queue.put(job)
        else:
            await self.normal_priority_queue.put(job)
```

## 🔍 **Troubleshooting**

### Common Issues

1. **Jobs stuck in queue**
   ```bash
   # Check worker status
   curl http://localhost:8000/workers/status
   
   # Check Redis queue length
   redis-cli llen ai:training:queue
   redis-cli llen ai:inference:queue
   ```

2. **GPU memory issues**
   ```bash
   # Monitor GPU usage
   nvidia-smi
   
   # Reduce concurrent jobs
   MAX_CONCURRENT_JOBS=1 docker compose up -d ai-worker-gpu
   ```

3. **CPU overload**
   ```bash
   # Monitor CPU usage
   htop
   
   # Scale workers
   docker compose up -d --scale ai-worker=4
   ```

### Performance Tuning

```bash
# Benchmark different configurations
for jobs in 1 2 4 8; do
    MAX_CONCURRENT_JOBS=$jobs python benchmark.py
done

# Monitor resource usage
docker stats ai-worker ai-worker-gpu
```

## 📈 **Scaling Strategies**

### Horizontal Scaling
- **Add more workers**: `docker compose up -d --scale ai-worker=10`
- **Distribute across machines**: Use Kubernetes or Docker Swarm
- **Load balancing**: Use Redis pub/sub for job distribution

### Vertical Scaling
- **Increase worker limits**: `MAX_CONCURRENT_JOBS=8`
- **Add GPU resources**: Use multi-GPU setups
- **Optimize models**: Use model quantization and optimization

### Hybrid Scaling
- **CPU workers for inference**: High throughput, low latency
- **GPU workers for training**: High performance, resource-intensive
- **Specialized workers**: Different workers for different model types
