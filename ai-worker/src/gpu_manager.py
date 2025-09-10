import asyncio
import logging
import subprocess
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class GPUStatus(Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    OOM = "out_of_memory"
    ERROR = "error"


@dataclass
class GPUInfo:
    device_id: int
    total_memory: int  # MB
    used_memory: int  # MB
    free_memory: int  # MB
    utilization: int  # %
    status: GPUStatus
    temperature: int  # Celsius


class GPUMemoryManager:
    """Manages GPU memory and prevents OOM situations."""

    def __init__(self, safety_margin_mb: int = 1000):
        self.safety_margin_mb = safety_margin_mb  # Reserve 1GB for safety
        self.gpu_cache = {}
        self.cache_ttl = 5  # Cache GPU info for 5 seconds
        self.last_update = 0

    async def get_gpu_info(self, device_id: int = 0) -> Optional[GPUInfo]:
        """Get current GPU information."""
        current_time = time.time()

        # Return cached info if recent
        if (
            device_id in self.gpu_cache
            and current_time - self.last_update < self.cache_ttl
        ):
            return self.gpu_cache[device_id]

        try:
            # Run nvidia-smi to get GPU info
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(device_id),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                logger.error(f"Failed to get GPU info: {result.stderr}")
                return None

            # Parse nvidia-smi output
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 6:
                gpu_info = GPUInfo(
                    device_id=int(parts[0]),
                    total_memory=int(parts[1]),
                    used_memory=int(parts[2]),
                    free_memory=int(parts[3]),
                    utilization=int(parts[4]),
                    temperature=int(parts[5]),
                    status=GPUStatus.AVAILABLE,
                )

                # Check for potential OOM conditions
                if gpu_info.free_memory < self.safety_margin_mb:
                    gpu_info.status = GPUStatus.OOM
                elif gpu_info.utilization > 90:
                    gpu_info.status = GPUStatus.BUSY

                # Cache the result
                self.gpu_cache[device_id] = gpu_info
                self.last_update = current_time

                return gpu_info

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout getting GPU info for device {device_id}")
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")

        return None

    async def can_allocate_memory(
        self, required_memory_mb: int, device_id: int = 0
    ) -> bool:
        """Check if we can safely allocate memory on GPU."""
        gpu_info = await self.get_gpu_info(device_id)
        if not gpu_info:
            return False

        # Check if we have enough free memory
        available_memory = gpu_info.free_memory - self.safety_margin_mb
        return available_memory >= required_memory_mb

    async def estimate_model_memory(self, model_type: str, batch_size: int = 1) -> int:
        """Estimate memory requirements for different model types."""
        # Rough estimates in MB
        base_memory = {
            "transformer": 2000,  # 2GB base
            "cnn": 1000,  # 1GB base
            "lstm": 500,  # 500MB base
            "bert": 3000,  # 3GB base
            "gpt": 4000,  # 4GB base
            "resnet": 800,  # 800MB base
            "vgg": 1500,  # 1.5GB base
        }

        base = base_memory.get(model_type.lower(), 1000)

        # Scale with batch size (rough approximation)
        memory_multiplier = 1 + (batch_size - 1) * 0.3

        return int(base * memory_multiplier)

    async def wait_for_gpu_memory(
        self, required_memory_mb: int, device_id: int = 0, timeout_seconds: int = 300
    ) -> bool:
        """Wait for GPU memory to become available."""
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            if await self.can_allocate_memory(required_memory_mb, device_id):
                return True

            logger.info(
                f"Waiting for GPU memory. Need {required_memory_mb}MB, "
                f"will retry in 10 seconds..."
            )
            await asyncio.sleep(10)

        logger.warning(
            f"Timeout waiting for GPU memory after {timeout_seconds} seconds"
        )
        return False

    async def cleanup_gpu_memory(self, device_id: int = 0) -> bool:
        """Attempt to free GPU memory by clearing cache."""
        try:
            # Clear PyTorch cache
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device_id)
                logger.info(f"Cleared GPU cache for device {device_id}")
                return True
        except Exception as e:
            logger.error(f"Error clearing GPU cache: {e}")

        return False

    async def get_optimal_batch_size(self, model_type: str, device_id: int = 0) -> int:
        """Calculate optimal batch size based on available GPU memory."""
        gpu_info = await self.get_gpu_info(device_id)
        if not gpu_info:
            return 1

        available_memory = gpu_info.free_memory - self.safety_margin_mb
        base_memory = await self.estimate_model_memory(model_type, batch_size=1)

        # Calculate how many batches we can fit
        max_batches = max(1, available_memory // base_memory)

        # Conservative approach: use half of max to be safe
        optimal_batch_size = max(1, max_batches // 2)

        logger.info(
            f"Optimal batch size for {model_type}: {optimal_batch_size} "
            f"(available: {available_memory}MB, base: {base_memory}MB)"
        )

        return optimal_batch_size


class JobQueueManager:
    """Manages job queues with GPU memory awareness."""

    def __init__(self, gpu_manager: GPUMemoryManager):
        self.gpu_manager = gpu_manager
        self.pending_jobs = []
        self.running_jobs = {}

    async def can_start_job(self, job_data: Dict) -> Tuple[bool, str]:
        """Check if a job can start based on GPU memory."""
        if not job_data.get("requires_gpu", False):
            return True, "CPU job"

        model_type = job_data.get("model_type", "transformer")
        batch_size = job_data.get("hyperparameters", {}).get("batch_size", 1)

        # Estimate memory requirements
        required_memory = await self.gpu_manager.estimate_model_memory(
            model_type, batch_size
        )

        # Check if we have enough memory
        device_id = job_data.get("gpu_device", 0)
        can_allocate = await self.gpu_manager.can_allocate_memory(
            required_memory, device_id
        )

        if can_allocate:
            return True, f"GPU memory available ({required_memory}MB required)"
        else:
            return False, f"Insufficient GPU memory ({required_memory}MB required)"

    async def optimize_job_parameters(self, job_data: Dict) -> Dict:
        """Optimize job parameters based on available GPU memory."""
        if not job_data.get("requires_gpu", False):
            return job_data

        model_type = job_data.get("model_type", "transformer")
        device_id = job_data.get("gpu_device", 0)

        # Get optimal batch size
        optimal_batch_size = await self.gpu_manager.get_optimal_batch_size(
            model_type, device_id
        )

        # Update job parameters
        optimized_job = job_data.copy()
        if "hyperparameters" not in optimized_job:
            optimized_job["hyperparameters"] = {}

        current_batch_size = optimized_job["hyperparameters"].get("batch_size", 1)
        if current_batch_size > optimal_batch_size:
            optimized_job["hyperparameters"]["batch_size"] = optimal_batch_size
            logger.info(
                f"Reduced batch size from {current_batch_size} to {optimal_batch_size}"
            )

        return optimized_job


