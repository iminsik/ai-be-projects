import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

class GPUStatus(Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    OUT_OF_MEMORY = "out_of_memory"
    ERROR = "error"

@dataclass
class GPUInfo:
    device_id: int
    total_memory: int  # MB
    used_memory: int   # MB
    free_memory: int   # MB
    utilization: float  # Percentage
    temperature: float  # Celsius
    status: GPUStatus

@dataclass
class JobMemoryRequirement:
    job_id: str
    estimated_memory: int  # MB
    priority: int  # Higher number = higher priority
    timeout: int  # seconds

class GPUMemoryManager:
    """Manages GPU memory allocation and prevents out-of-memory errors."""
    
    def __init__(self, safety_margin_mb: int = 1000):
        self.safety_margin_mb = safety_margin_mb
        self.device_memory_cache = {}
        self.pending_jobs = {}  # job_id -> JobMemoryRequirement
        self.active_jobs = {}   # job_id -> allocated_memory
        self.memory_lock = asyncio.Lock()
        
    async def get_gpu_info(self, device_id: int = 0) -> Optional[GPUInfo]:
        """Get current GPU information."""
        try:
            if not torch.cuda.is_available():
                return None
                
            device = torch.device(f"cuda:{device_id}")
            
            # Get memory info
            total_memory = torch.cuda.get_device_properties(device).total_memory
            total_memory_mb = total_memory // (1024 * 1024)
            
            # Get current memory usage
            torch.cuda.set_device(device)
            allocated_memory = torch.cuda.memory_allocated(device)
            cached_memory = torch.cuda.memory_reserved(device)
            used_memory_mb = (allocated_memory + cached_memory) // (1024 * 1024)
            free_memory_mb = total_memory_mb - used_memory_mb
            
            # Get utilization (simplified - in real implementation, use nvidia-ml-py)
            utilization = (used_memory_mb / total_memory_mb) * 100
            
            # Determine status
            if free_memory_mb < self.safety_margin_mb:
                status = GPUStatus.OUT_OF_MEMORY
            elif utilization > 90:
                status = GPUStatus.BUSY
            else:
                status = GPUStatus.AVAILABLE
                
            return GPUInfo(
                device_id=device_id,
                total_memory=total_memory_mb,
                used_memory=used_memory_mb,
                free_memory=free_memory_mb,
                utilization=utilization,
                temperature=0.0,  # Would need nvidia-ml-py for real temperature
                status=status
            )
            
        except Exception as e:
            logger.error(f"Error getting GPU info for device {device_id}: {e}")
            return None
    
    async def estimate_model_memory(self, model_type: str, batch_size: int, 
                                  sequence_length: int = 512) -> int:
        """Estimate memory requirements for a model."""
        # Memory estimation based on model type and parameters
        memory_estimates = {
            "transformer": {
                "small": 2000,   # 2GB
                "medium": 4000,  # 4GB
                "large": 8000,   # 8GB
                "xlarge": 16000  # 16GB
            },
            "cnn": {
                "small": 1000,   # 1GB
                "medium": 2000,  # 2GB
                "large": 4000,   # 4GB
            },
            "rnn": {
                "small": 1500,   # 1.5GB
                "medium": 3000,  # 3GB
                "large": 6000,   # 6GB
            }
        }
        
        # Base memory for model type
        base_memory = memory_estimates.get(model_type, {}).get("medium", 2000)
        
        # Scale by batch size
        batch_multiplier = max(1, batch_size / 8)  # Assume 8 is standard batch size
        
        # Scale by sequence length for transformers
        if model_type == "transformer":
            seq_multiplier = max(1, sequence_length / 512)
            base_memory *= seq_multiplier
        
        estimated_memory = int(base_memory * batch_multiplier)
        
        # Add overhead for gradients, optimizer states, etc.
        total_memory = int(estimated_memory * 2.5)  # 2.5x for training overhead
        
        logger.info(f"Estimated memory for {model_type} (batch={batch_size}): {total_memory}MB")
        return total_memory
    
    async def can_allocate_memory(self, required_memory: int, device_id: int = 0) -> Tuple[bool, str]:
        """Check if we can allocate the required memory."""
        async with self.memory_lock:
            gpu_info = await self.get_gpu_info(device_id)
            if not gpu_info:
                return False, "GPU not available"
            
            available_memory = gpu_info.free_memory - self.safety_margin_mb
            
            if available_memory >= required_memory:
                return True, f"Memory available: {available_memory}MB >= {required_memory}MB"
            else:
                return False, f"Insufficient memory: {available_memory}MB < {required_memory}MB"
    
    async def allocate_memory(self, job_id: str, required_memory: int, 
                            device_id: int = 0) -> bool:
        """Allocate memory for a job."""
        async with self.memory_lock:
            can_allocate, reason = await self.can_allocate_memory(required_memory, device_id)
            
            if can_allocate:
                self.active_jobs[job_id] = {
                    "allocated_memory": required_memory,
                    "device_id": device_id,
                    "allocated_at": time.time()
                }
                logger.info(f"Allocated {required_memory}MB for job {job_id}")
                return True
            else:
                logger.warning(f"Cannot allocate memory for job {job_id}: {reason}")
                return False
    
    async def deallocate_memory(self, job_id: str):
        """Deallocate memory for a job."""
        async with self.memory_lock:
            if job_id in self.active_jobs:
                allocated_memory = self.active_jobs[job_id]["allocated_memory"]
                del self.active_jobs[job_id]
                logger.info(f"Deallocated {allocated_memory}MB for job {job_id}")
                
                # Clear GPU cache to free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    async def wait_for_memory(self, required_memory: int, device_id: int = 0, 
                            timeout: int = 300) -> bool:
        """Wait for memory to become available."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            can_allocate, reason = await self.can_allocate_memory(required_memory, device_id)
            
            if can_allocate:
                return True
            
            logger.info(f"Waiting for memory: {reason}")
            await asyncio.sleep(5)  # Check every 5 seconds
        
        return False
    
    async def optimize_job_parameters(self, job_data: Dict) -> Dict:
        """Optimize job parameters to fit available memory."""
        model_type = job_data.get("model_type", "transformer")
        hyperparameters = job_data.get("hyperparameters", {})
        
        # Get current GPU info
        gpu_info = await self.get_gpu_info()
        if not gpu_info:
            return job_data
        
        available_memory = gpu_info.free_memory - self.safety_margin_mb
        
        # Try to optimize batch size
        current_batch_size = hyperparameters.get("batch_size", 8)
        optimized_batch_size = current_batch_size
        
        # Reduce batch size until it fits
        while optimized_batch_size > 1:
            estimated_memory = await self.estimate_model_memory(
                model_type, optimized_batch_size
            )
            
            if estimated_memory <= available_memory:
                break
                
            optimized_batch_size = max(1, optimized_batch_size // 2)
        
        # Update hyperparameters
        optimized_hyperparameters = hyperparameters.copy()
        optimized_hyperparameters["batch_size"] = optimized_batch_size
        
        # If we had to reduce batch size significantly, suggest gradient accumulation
        if optimized_batch_size < current_batch_size:
            gradient_accumulation_steps = current_batch_size // optimized_batch_size
            optimized_hyperparameters["gradient_accumulation_steps"] = gradient_accumulation_steps
            logger.info(f"Optimized batch size: {current_batch_size} -> {optimized_batch_size}")
            logger.info(f"Added gradient accumulation steps: {gradient_accumulation_steps}")
        
        # Create optimized job data
        optimized_job_data = job_data.copy()
        optimized_job_data["hyperparameters"] = optimized_hyperparameters
        optimized_job_data["optimized"] = True
        optimized_job_data["original_batch_size"] = current_batch_size
        
        return optimized_job_data
    
    async def get_memory_usage_summary(self) -> Dict:
        """Get summary of current memory usage."""
        async with self.memory_lock:
            total_allocated = sum(job["allocated_memory"] for job in self.active_jobs.values())
            pending_requirements = sum(job.estimated_memory for job in self.pending_jobs.values())
            
            return {
                "active_jobs": len(self.active_jobs),
                "pending_jobs": len(self.pending_jobs),
                "total_allocated_mb": total_allocated,
                "pending_requirements_mb": pending_requirements,
                "active_job_details": {
                    job_id: {
                        "allocated_memory": job["allocated_memory"],
                        "device_id": job["device_id"],
                        "allocated_at": job["allocated_at"]
                    }
                    for job_id, job in self.active_jobs.items()
                }
            }

class JobQueueManager:
    """Manages job queuing with memory-aware scheduling."""
    
    def __init__(self, gpu_manager: GPUMemoryManager):
        self.gpu_manager = gpu_manager
        self.job_queue = asyncio.Queue()
        self.priority_queue = asyncio.Queue()
        
    async def can_start_job(self, job_data: Dict) -> Tuple[bool, str]:
        """Check if a job can start immediately."""
        model_type = job_data.get("model_type", "transformer")
        hyperparameters = job_data.get("hyperparameters", {})
        batch_size = hyperparameters.get("batch_size", 8)
        
        required_memory = await self.gpu_manager.estimate_model_memory(
            model_type, batch_size
        )
        
        return await self.gpu_manager.can_allocate_memory(required_memory)
    
    async def queue_job(self, job_data: Dict, priority: int = 0):
        """Queue a job for execution."""
        job_id = job_data["job_id"]
        model_type = job_data.get("model_type", "transformer")
        hyperparameters = job_data.get("hyperparameters", {})
        batch_size = hyperparameters.get("batch_size", 8)
        
        required_memory = await self.gpu_manager.estimate_model_memory(
            model_type, batch_size
        )
        
        job_requirement = JobMemoryRequirement(
            job_id=job_id,
            estimated_memory=required_memory,
            priority=priority,
            timeout=300
        )
        
        if priority > 0:
            await self.priority_queue.put(job_requirement)
        else:
            await self.job_queue.put(job_requirement)
        
        self.gpu_manager.pending_jobs[job_id] = job_requirement
        logger.info(f"Queued job {job_id} with {required_memory}MB requirement")
    
    async def get_next_job(self) -> Optional[JobMemoryRequirement]:
        """Get the next job that can be executed."""
        # Try priority queue first
        if not self.priority_queue.empty():
            job_requirement = await self.priority_queue.get()
            can_allocate, _ = await self.gpu_manager.can_allocate_memory(
                job_requirement.estimated_memory
            )
            if can_allocate:
                return job_requirement
            else:
                # Put it back and try regular queue
                await self.priority_queue.put(job_requirement)
        
        # Try regular queue
        if not self.job_queue.empty():
            job_requirement = await self.job_queue.get()
            can_allocate, _ = await self.gpu_manager.can_allocate_memory(
                job_requirement.estimated_memory
            )
            if can_allocate:
                return job_requirement
            else:
                # Put it back
                await self.job_queue.put(job_requirement)
        
        return None
