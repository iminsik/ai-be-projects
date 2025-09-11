import asyncio
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Worker configuration
WORKER_CONFIGS = {
    "pytorch-2.0": {
        "dockerfile": "Dockerfile.pytorch-2.0",
        "framework": "pytorch",
        "version": "2.0.0",
        "gpu": False,
        "startup_time": 30,  # seconds
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
        "idle_timeout": 600,  # 10 minutes for GPU workers
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


@dataclass
class WorkerInstance:
    worker_id: str
    worker_type: str
    container_name: str
    status: str  # starting, running, stopping, stopped
    created_at: datetime
    last_activity: datetime
    job_count: int = 0
    max_jobs: int = 2


class WorkerManager:
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.active_workers: Dict[str, WorkerInstance] = {}
        self.worker_queues: Dict[str, str] = {
            "pytorch-2.0": "ai:training:pytorch-2.0:queue",
            "pytorch-2.1": "ai:training:pytorch-2.1:queue",
            "pytorch-2.0-gpu": "ai:training:pytorch-2.0-gpu:queue",
            "pytorch-2.1-gpu": "ai:training:pytorch-2.1-gpu:queue",
            "tensorflow": "ai:training:tensorflow:queue",
            "sklearn": "ai:training:sklearn:queue",
        }
        self.cleanup_task: Optional[asyncio.Task] = None

    async def connect_redis(self):
        """Connect to Redis."""
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            decode_responses=True,
        )
        await self.redis_client.ping()
        logger.info("Connected to Redis")

    async def start_worker(self, worker_type: str) -> str:
        """Start a new worker instance."""
        if worker_type not in WORKER_CONFIGS:
            raise ValueError(f"Unknown worker type: {worker_type}")

        config = WORKER_CONFIGS[worker_type]
        worker_id = f"{worker_type}-{int(time.time())}"
        container_name = f"ai-worker-{worker_id}"

        # Check if we already have enough workers of this type
        active_count = sum(
            1
            for w in self.active_workers.values()
            if w.worker_type == worker_type and w.status == "running"
        )

        if active_count >= 3:  # Max 3 workers per type
            logger.warning(f"Maximum workers reached for {worker_type}")
            return None

        logger.info(f"Starting worker {worker_id} of type {worker_type}")

        # Create worker instance
        worker = WorkerInstance(
            worker_id=worker_id,
            worker_type=worker_type,
            container_name=container_name,
            status="starting",
            created_at=datetime.now(),
            last_activity=datetime.now(),
        )
        self.active_workers[worker_id] = worker

        # Start Docker container
        try:
            await self._start_docker_worker(worker, config)
            worker.status = "running"
            logger.info(f"Worker {worker_id} started successfully")
            return worker_id
        except Exception as e:
            worker.status = "stopped"
            logger.error(f"Failed to start worker {worker_id}: {e}")
            return None

    async def _start_docker_worker(self, worker: WorkerInstance, config: Dict):
        """Start a Docker container for the worker."""
        env_vars = {
            "REDIS_HOST": os.getenv("REDIS_HOST", "redis"),
            "REDIS_PORT": os.getenv("REDIS_PORT", "6379"),
            "REDIS_DB": os.getenv("REDIS_DB", "0"),
            "MODEL_FRAMEWORK": config["framework"],
            "MODEL_VERSION": config["version"],
            "WORKER_TYPE": worker.worker_type,
        }

        if config["gpu"]:
            env_vars["USE_GPU"] = "true"

        # Build Docker command
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            worker.container_name,
            "--network",
            "ai-job-queue-network",
            "--restart",
            "unless-stopped",
        ]

        # Add environment variables
        for key, value in env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Add volumes
        cmd.extend(
            [
                "-v",
                f"{os.getcwd()}/ai-worker:/app",
                "-v",
                "/app/.venv",
                "-v",
                "ai_models:/app/models",
            ]
        )

        # Add GPU support if needed
        if config["gpu"]:
            cmd.extend(["--runtime", "nvidia", "-e", "NVIDIA_VISIBLE_DEVICES=all"])

        # Add image name
        cmd.append(f"ai-worker-{worker.worker_type}")

        # Execute command
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Docker command failed: {result.stderr}")

        # Wait for worker to be ready
        await asyncio.sleep(config["startup_time"])

    async def stop_worker(self, worker_id: str) -> bool:
        """Stop a worker instance."""
        if worker_id not in self.active_workers:
            return False

        worker = self.active_workers[worker_id]
        logger.info(f"Stopping worker {worker_id}")

        try:
            # Stop Docker container
            result = subprocess.run(
                ["docker", "stop", worker.container_name],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                # Remove container
                subprocess.run(
                    ["docker", "rm", worker.container_name],
                    capture_output=True,
                    text=True,
                )
                worker.status = "stopped"
                logger.info(f"Worker {worker_id} stopped successfully")
                return True
            else:
                logger.error(f"Failed to stop worker {worker_id}: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error stopping worker {worker_id}: {e}")
            return False

    async def get_available_worker(self, worker_type: str) -> Optional[str]:
        """Get an available worker of the specified type."""
        available_workers = [
            worker_id
            for worker_id, worker in self.active_workers.items()
            if (
                worker.worker_type == worker_type
                and worker.status == "running"
                and worker.job_count < worker.max_jobs
            )
        ]

        if available_workers:
            # Return the least busy worker
            return min(
                available_workers, key=lambda w_id: self.active_workers[w_id].job_count
            )
        return None

    async def ensure_worker_available(self, worker_type: str) -> Optional[str]:
        """Ensure at least one worker is available for the specified type."""
        # Check if we have an available worker
        worker_id = await self.get_available_worker(worker_type)
        if worker_id:
            return worker_id

        # Start a new worker if none available
        logger.info(f"No available workers for {worker_type}, starting new one")
        return await self.start_worker(worker_type)

    async def update_worker_activity(self, worker_id: str):
        """Update the last activity time for a worker."""
        if worker_id in self.active_workers:
            self.active_workers[worker_id].last_activity = datetime.now()

    async def cleanup_idle_workers(self):
        """Clean up idle workers."""
        current_time = datetime.now()
        idle_workers = []

        for worker_id, worker in self.active_workers.items():
            if worker.status != "running":
                continue

            config = WORKER_CONFIGS[worker.worker_type]
            idle_time = current_time - worker.last_activity

            if (
                idle_time > timedelta(seconds=config["idle_timeout"])
                and worker.job_count == 0
            ):
                idle_workers.append(worker_id)

        for worker_id in idle_workers:
            logger.info(f"Cleaning up idle worker {worker_id}")
            await self.stop_worker(worker_id)

    async def get_worker_status(self) -> Dict:
        """Get status of all workers."""
        return {
            "total_workers": len(self.active_workers),
            "active_workers": len(
                [w for w in self.active_workers.values() if w.status == "running"]
            ),
            "workers_by_type": {
                worker_type: len(
                    [
                        w
                        for w in self.active_workers.values()
                        if w.worker_type == worker_type and w.status == "running"
                    ]
                )
                for worker_type in WORKER_CONFIGS.keys()
            },
            "worker_details": {
                worker_id: asdict(worker)
                for worker_id, worker in self.active_workers.items()
            },
        }

    async def start_cleanup_task(self):
        """Start the background cleanup task."""

        async def cleanup_loop():
            while True:
                try:
                    await self.cleanup_idle_workers()
                    await asyncio.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Error in cleanup loop: {e}")
                    await asyncio.sleep(60)

        self.cleanup_task = asyncio.create_task(cleanup_loop())


# Global worker manager instance
worker_manager = WorkerManager()

# FastAPI app
app = FastAPI(
    title="Worker Manager API",
    description="API for managing dynamic AI workers",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    """Initialize the worker manager."""
    await worker_manager.connect_redis()
    await worker_manager.start_cleanup_task()
    logger.info("Worker Manager started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    if worker_manager.cleanup_task:
        worker_manager.cleanup_task.cancel()
    logger.info("Worker Manager stopped")


@app.post("/workers/start/{worker_type}")
async def start_worker_endpoint(worker_type: str):
    """Start a new worker of the specified type."""
    worker_id = await worker_manager.start_worker(worker_type)
    if worker_id:
        return {"worker_id": worker_id, "status": "started"}
    else:
        raise HTTPException(status_code=400, detail="Failed to start worker")


@app.post("/workers/stop/{worker_id}")
async def stop_worker_endpoint(worker_id: str):
    """Stop a specific worker."""
    success = await worker_manager.stop_worker(worker_id)
    if success:
        return {"status": "stopped"}
    else:
        raise HTTPException(status_code=404, detail="Worker not found")


@app.get("/workers/status")
async def get_worker_status_endpoint():
    """Get status of all workers."""
    return await worker_manager.get_worker_status()


@app.post("/workers/ensure/{worker_type}")
async def ensure_worker_endpoint(worker_type: str):
    """Ensure a worker is available for the specified type."""
    worker_id = await worker_manager.ensure_worker_available(worker_type)
    if worker_id:
        return {"worker_id": worker_id, "status": "available"}
    else:
        raise HTTPException(
            status_code=500, detail="Failed to ensure worker availability"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
