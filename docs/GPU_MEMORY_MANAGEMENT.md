# GPU Memory Management Guide

## Overview

This guide explains how the AI worker system gracefully handles GPU memory constraints to prevent out-of-memory errors during training and inference.

## 🚀 **Key Features**

### 1. **Memory Estimation**
- Automatically estimates memory requirements for different model types
- Considers batch size, sequence length, and model architecture
- Includes overhead for gradients, optimizer states, and activations

### 2. **Memory Allocation**
- Pre-allocates memory before starting jobs
- Tracks allocated memory per job
- Prevents multiple jobs from exceeding GPU capacity

### 3. **Graceful Degradation**
- Automatically reduces batch size when memory is insufficient
- Implements gradient accumulation to maintain effective batch size
- Queues jobs when memory is unavailable

### 4. **Memory Monitoring**
- Real-time GPU memory usage tracking
- Memory utilization alerts
- Automatic cleanup on job completion/failure

## 🔧 **Configuration**

### Environment Variables
```bash
# Enable GPU memory management
USE_GPU=true
GPU_DEVICE=0

# Memory management settings
MAX_CONCURRENT_JOBS=2  # Reduce for GPU jobs
JOB_TIMEOUT=3600       # Timeout for memory allocation
```

### Docker Compose GPU Setup
```yaml
services:
  ai-worker-gpu:
    build:
      context: ./ai-worker
      dockerfile: Dockerfile.gpu
    environment:
      - USE_GPU=true
      - GPU_DEVICE=0
      - MAX_CONCURRENT_JOBS=1  # Conservative for GPU
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 📊 **Memory Estimation**

### Model Type Estimates
```python
# Memory requirements (MB) for different model types
memory_estimates = {
    "transformer": {
        "small": 2000,   # 2GB - BERT-base
        "medium": 4000,  # 4GB - BERT-large
        "large": 8000,   # 8GB - GPT-2
        "xlarge": 16000  # 16GB - GPT-3
    },
    "cnn": {
        "small": 1000,   # 1GB - ResNet-50
        "medium": 2000,  # 2GB - ResNet-101
        "large": 4000,   # 4GB - ResNet-152
    },
    "rnn": {
        "small": 1500,   # 1.5GB - LSTM
        "medium": 3000,  # 3GB - GRU
        "large": 6000,   # 6GB - Transformer RNN
    }
}
```

### Batch Size Scaling
```python
# Memory scales with batch size
base_memory = 2000  # MB for batch_size=8
batch_size = 32
scaled_memory = base_memory * (batch_size / 8)  # 8000 MB

# Training overhead (gradients, optimizer states)
total_memory = scaled_memory * 2.5  # 20000 MB
```

## 🎯 **Graceful Handling Strategies**

### 1. **Automatic Batch Size Reduction**
```python
# When memory is insufficient, automatically reduce batch size
original_batch_size = 32
available_memory = 8000  # MB

# Reduce batch size until it fits
optimized_batch_size = 8  # Reduced to fit memory

# Use gradient accumulation to maintain effective batch size
gradient_accumulation_steps = 32 // 8  # 4 steps
effective_batch_size = 8 * 4  # Still 32
```

### 2. **Job Queuing**
```python
# Jobs wait in queue when memory is unavailable
async def process_training_job(job_data):
    required_memory = estimate_memory(job_data)
    
    # Check if memory is available
    can_allocate, reason = await gpu_manager.can_allocate_memory(required_memory)
    
    if not can_allocate:
        # Wait for memory to become available
        memory_available = await gpu_manager.wait_for_memory(
            required_memory, timeout=300
        )
        
        if not memory_available:
            # Job fails gracefully with clear error message
            await update_job_status(job_id, "failed", 
                                  error="GPU memory timeout")
            return
```

### 3. **Memory Optimization**
```python
# Optimize job parameters for available memory
async def optimize_job_parameters(job_data):
    gpu_info = await gpu_manager.get_gpu_info()
    available_memory = gpu_info.free_memory - safety_margin
    
    # Reduce batch size
    while batch_size > 1:
        estimated_memory = estimate_memory(model_type, batch_size)
        if estimated_memory <= available_memory:
            break
        batch_size = max(1, batch_size // 2)
    
    # Add gradient accumulation
    if batch_size < original_batch_size:
        gradient_accumulation_steps = original_batch_size // batch_size
        hyperparameters["gradient_accumulation_steps"] = gradient_accumulation_steps
    
    return optimized_job_data
```

## 📈 **Monitoring and Alerts**

### GPU Memory Status
```python
# Get real-time GPU memory information
gpu_info = await gpu_manager.get_gpu_info()
print(f"Total Memory: {gpu_info.total_memory}MB")
print(f"Used Memory: {gpu_info.used_memory}MB")
print(f"Free Memory: {gpu_info.free_memory}MB")
print(f"Utilization: {gpu_info.utilization}%")
print(f"Status: {gpu_info.status}")
```

### Memory Usage Summary
```python
# Get memory usage summary
memory_summary = await gpu_manager.get_memory_usage_summary()
print(f"Active Jobs: {memory_summary['active_jobs']}")
print(f"Pending Jobs: {memory_summary['pending_jobs']}")
print(f"Total Allocated: {memory_summary['total_allocated_mb']}MB")
print(f"Pending Requirements: {memory_summary['pending_requirements_mb']}MB")
```

## 🔍 **Error Handling**

### Common GPU Memory Errors
```python
# 1. Out of Memory Error
try:
    model = load_large_model()
except torch.cuda.OutOfMemoryError:
    # Automatically reduce batch size and retry
    optimized_params = await gpu_manager.optimize_job_parameters(job_data)
    model = load_model_with_optimized_params(optimized_params)

# 2. Memory Allocation Failure
can_allocate, reason = await gpu_manager.can_allocate_memory(required_memory)
if not can_allocate:
    logger.warning(f"Cannot allocate memory: {reason}")
    # Queue job for later execution
    await job_queue_manager.queue_job(job_data)

# 3. Memory Timeout
memory_available = await gpu_manager.wait_for_memory(required_memory, timeout=300)
if not memory_available:
    error_msg = f"GPU memory timeout after 5 minutes"
    await update_job_status(job_id, "failed", error=error_msg)
```

## 🚀 **Best Practices**

### 1. **Conservative Memory Limits**
```bash
# Set conservative limits for GPU workers
MAX_CONCURRENT_JOBS=1  # One job per GPU
SAFETY_MARGIN_MB=1000  # 1GB safety margin
```

### 2. **Progressive Batch Size**
```python
# Start with small batch sizes and scale up
batch_sizes = [1, 2, 4, 8, 16, 32]
for batch_size in batch_sizes:
    try:
        model = train_with_batch_size(batch_size)
        break
    except torch.cuda.OutOfMemoryError:
        continue
```

### 3. **Memory Cleanup**
```python
# Always cleanup memory after job completion
try:
    result = await process_job(job_data)
finally:
    await gpu_manager.deallocate_memory(job_id)
    torch.cuda.empty_cache()
```

### 4. **Monitoring and Alerts**
```python
# Set up memory monitoring
async def monitor_gpu_memory():
    while True:
        gpu_info = await gpu_manager.get_gpu_info()
        if gpu_info.utilization > 90:
            logger.warning(f"GPU memory utilization high: {gpu_info.utilization}%")
        await asyncio.sleep(30)
```

## 📊 **Performance Optimization**

### 1. **Memory Pooling**
```python
# Reuse memory allocations when possible
class MemoryPool:
    def __init__(self):
        self.available_chunks = {}
    
    async def get_memory_chunk(self, size):
        if size in self.available_chunks:
            return self.available_chunks[size].pop()
        return torch.cuda.FloatTensor(size)
    
    async def return_memory_chunk(self, size, chunk):
        if size not in self.available_chunks:
            self.available_chunks[size] = []
        self.available_chunks[size].append(chunk)
```

### 2. **Gradient Checkpointing**
```python
# Use gradient checkpointing to reduce memory usage
from torch.utils.checkpoint import checkpoint

class MemoryEfficientModel(nn.Module):
    def forward(self, x):
        return checkpoint(self._forward, x)
    
    def _forward(self, x):
        # Forward pass without storing intermediate activations
        pass
```

### 3. **Mixed Precision Training**
```python
# Use mixed precision to reduce memory usage
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 🔧 **Troubleshooting**

### Common Issues and Solutions

1. **Jobs stuck in queue**
   ```bash
   # Check GPU memory status
   nvidia-smi
   
   # Check worker memory summary
   curl http://localhost:8000/workers/status
   ```

2. **Memory allocation failures**
   ```bash
   # Reduce concurrent jobs
   MAX_CONCURRENT_JOBS=1 docker compose up -d ai-worker-gpu
   
   # Increase safety margin
   SAFETY_MARGIN_MB=2000 docker compose up -d ai-worker-gpu
   ```

3. **Out of memory errors**
   ```bash
   # Monitor memory usage
   watch -n 1 nvidia-smi
   
   # Clear GPU cache
   python -c "import torch; torch.cuda.empty_cache()"
   ```

This comprehensive GPU memory management system ensures that your AI training jobs run smoothly without running into out-of-memory errors, while maximizing GPU utilization through intelligent resource allocation and graceful degradation strategies.
