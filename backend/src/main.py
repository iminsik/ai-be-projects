import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import redis.asyncio as redis
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from .config import Config
except ImportError:
    # Fallback for direct execution
    from config import Config

# Initialize FastAPI app
app = FastAPI(
    title="AI Job Queue API",
    description="API for submitting and monitoring AI training and inference jobs",
    version="1.0.0",
)

# Redis connection
redis_client: Optional[redis.Redis] = None

# Worker Manager configuration
WORKER_MANAGER_URL = os.getenv("WORKER_MANAGER_URL", "http://localhost:8001")
WORKER_MANAGER_TYPE = os.getenv("WORKER_MANAGER_TYPE", "local")  # "docker" or "local"

# Model framework registry
MODEL_FRAMEWORK_REGISTRY = {
    "bert": {
        "framework": "pytorch",
        "version": "2.1.0",
        "queue": "ai:training:pytorch-2.1:queue",
    },
    "gpt": {
        "framework": "pytorch",
        "version": "2.1.0",
        "queue": "ai:training:pytorch-2.1:queue",
    },
    "resnet": {
        "framework": "pytorch",
        "version": "2.0.0",
        "queue": "ai:training:pytorch-2.0:queue",
    },
    "vgg": {
        "framework": "pytorch",
        "version": "2.0.0",
        "queue": "ai:training:pytorch-2.0:queue",
    },
    "inception": {
        "framework": "tensorflow",
        "version": "2.13.0",
        "queue": "ai:training:tensorflow:queue",
    },
    "mobilenet": {
        "framework": "tensorflow",
        "version": "2.13.0",
        "queue": "ai:training:tensorflow:queue",
    },
    "random_forest": {
        "framework": "sklearn",
        "version": "1.3.0",
        "queue": "ai:training:sklearn:queue",
    },
    "svm": {
        "framework": "sklearn",
        "version": "1.3.0",
        "queue": "ai:training:sklearn:queue",
    },
    "logistic_regression": {
        "framework": "sklearn",
        "version": "1.3.0",
        "queue": "ai:training:sklearn:queue",
    },
}


# Pydantic models
class TrainingJobRequest(BaseModel):
    model_type: str = Field(..., description="Type of model to train")
    data_path: str = Field(..., description="Path to training data")
    hyperparameters: Optional[Dict] = Field(
        default_factory=dict, description="Training hyperparameters"
    )
    description: Optional[str] = Field(None, description="Job description")
    requires_gpu: Optional[bool] = Field(False, description="Whether GPU is required")
    framework_override: Optional[str] = Field(
        None, description="Override framework selection"
    )


class InferenceJobRequest(BaseModel):
    model_id: str = Field(..., description="ID of the trained model")
    input_data: str = Field(..., description="Input data for inference")
    parameters: Optional[Dict] = Field(
        default_factory=dict, description="Inference parameters"
    )


class JobStatus(BaseModel):
    job_id: str
    job_type: str
    status: str  # pending, running, completed, failed
    created_at: datetime
    updated_at: datetime
    result: Optional[Dict] = None
    error: Optional[str] = None
    metadata: Optional[Dict] = None
    framework: Optional[str] = None
    worker_type: Optional[str] = None


# Redis keys
TRAINING_QUEUE = "ai:training:queue"
INFERENCE_QUEUE = "ai:inference:queue"
JOB_STATUS_PREFIX = "ai:job:status:"

# Framework-specific queues
FRAMEWORK_QUEUES = {
    "pytorch-2.0": "ai:training:pytorch-2.0:queue",
    "pytorch-2.1": "ai:training:pytorch-2.1:queue",
    "tensorflow": "ai:training:tensorflow:queue",
    "sklearn": "ai:training:sklearn:queue",
    "pytorch-2.0-gpu": "ai:training:pytorch-2.0-gpu:queue",
    "pytorch-2.1-gpu": "ai:training:pytorch-2.1-gpu:queue",
}


@app.on_event("startup")
async def startup_event():
    """Initialize Redis connection on startup."""
    global redis_client
    redis_client = redis.Redis(
        host=Config.REDIS_HOST,
        port=Config.REDIS_PORT,
        db=Config.REDIS_DB,
        password=Config.REDIS_PASSWORD,
        decode_responses=True,
    )
    await redis_client.ping()


@app.on_event("shutdown")
async def shutdown_event():
    """Close Redis connection on shutdown."""
    if redis_client:
        await redis_client.close()


async def get_redis() -> redis.Redis:
    """Get Redis client instance."""
    if redis_client is None:
        raise HTTPException(status_code=500, detail="Redis not initialized")
    return redis_client


async def ensure_worker_available(worker_type: str) -> bool:
    """Ensure a worker is available for the specified type."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # First, check if workers are already available
            status_response = await client.get(f"{WORKER_MANAGER_URL}/workers/status")
            if status_response.status_code == 200:
                status_data = status_response.json()
                workers_by_type = status_data.get("workers_by_type", {})

                # Check if we have available workers of this type
                if workers_by_type.get(worker_type, 0) > 0:
                    logger.info(f"Worker type {worker_type} is already available")
                    return True

            # No workers available, spawn a new one
            logger.info(f"Spawning new worker of type: {worker_type}")
            spawn_response = await client.post(
                f"{WORKER_MANAGER_URL}/workers/ensure/{worker_type}"
            )

            if spawn_response.status_code != 200:
                logger.error(
                    f"Failed to spawn worker {worker_type}: {spawn_response.text}"
                )
                return False

            # Wait for the worker to become available (polling)
            logger.info(f"Waiting for worker {worker_type} to become available...")
            max_wait_time = 60  # Maximum wait time in seconds
            poll_interval = 2  # Poll every 2 seconds
            waited_time = 0

            while waited_time < max_wait_time:
                await asyncio.sleep(poll_interval)
                waited_time += poll_interval

                # Check if worker is now available
                status_response = await client.get(
                    f"{WORKER_MANAGER_URL}/workers/status"
                )
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    workers_by_type = status_data.get("workers_by_type", {})

                    if workers_by_type.get(worker_type, 0) > 0:
                        logger.info(
                            f"Worker {worker_type} is now available after {waited_time}s"
                        )
                        return True

                logger.info(
                    f"Still waiting for worker {worker_type}... ({waited_time}s)"
                )

            logger.error(
                f"Timeout waiting for worker {worker_type} to become available"
            )
            return False

    except httpx.TimeoutException:
        logger.error(f"Timeout while ensuring worker {worker_type} is available")
        return False
    except Exception as e:
        logger.error(f"Error ensuring worker {worker_type} is available: {str(e)}")
        return False


def determine_framework_and_queue(
    model_type: str,
    requires_gpu: bool = False,
    framework_override: Optional[str] = None,
) -> tuple[str, str]:
    """Determine the appropriate framework and queue for a model type."""
    if framework_override:
        # Use override if provided
        if framework_override in FRAMEWORK_QUEUES:
            return framework_override, FRAMEWORK_QUEUES[framework_override]
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid framework override: {framework_override}. Available: {list(FRAMEWORK_QUEUES.keys())}",
            )

    # Look up model type in registry
    if model_type.lower() in MODEL_FRAMEWORK_REGISTRY:
        model_info = MODEL_FRAMEWORK_REGISTRY[model_type.lower()]
        framework = model_info["framework"]
        version = model_info["version"]

        # Determine if GPU version should be used
        if requires_gpu and framework == "pytorch":
            worker_type = f"{framework}-{version}-gpu"
        else:
            worker_type = f"{framework}-{version}"

        if worker_type in FRAMEWORK_QUEUES:
            return worker_type, FRAMEWORK_QUEUES[worker_type]

    # Default fallback
    if requires_gpu:
        return "pytorch-2.1-gpu", FRAMEWORK_QUEUES["pytorch-2.1-gpu"]
    else:
        return "pytorch-2.1", FRAMEWORK_QUEUES["pytorch-2.1"]


async def create_job_status(
    job_id: str,
    job_type: str,
    metadata: Dict,
    framework: str = None,
    worker_type: str = None,
) -> JobStatus:
    """Create a new job status entry."""
    now = datetime.utcnow()
    job_status = JobStatus(
        job_id=job_id,
        job_type=job_type,
        status="pending",
        created_at=now,
        updated_at=now,
        metadata=metadata,
        framework=framework,
        worker_type=worker_type,
    )

    redis_client = await get_redis()
    await redis_client.set(
        f"{JOB_STATUS_PREFIX}{job_id}",
        job_status.model_dump_json(),
        ex=86400,  # Expire after 24 hours
    )

    return job_status


async def update_job_status(
    job_id: str, status: str, result: Optional[Dict] = None, error: Optional[str] = None
):
    """Update job status in Redis."""
    redis_client = await get_redis()
    status_key = f"{JOB_STATUS_PREFIX}{job_id}"

    status_data = await redis_client.get(status_key)
    if status_data:
        job_status = JobStatus.model_validate_json(status_data)
        job_status.status = status
        job_status.updated_at = datetime.utcnow()
        if result is not None:
            job_status.result = result
        if error is not None:
            job_status.error = error

        await redis_client.set(status_key, job_status.model_dump_json(), ex=86400)


async def get_job_status(job_id: str) -> Optional[JobStatus]:
    """Get job status from Redis."""
    redis_client = await get_redis()
    status_data = await redis_client.get(f"{JOB_STATUS_PREFIX}{job_id}")

    if status_data:
        return JobStatus.model_validate_json(status_data)
    return None


@app.post("/jobs/training", response_model=JobStatus)
async def submit_training_job(request: TrainingJobRequest):
    """Submit a new training job to the appropriate framework-specific queue."""
    job_id = str(uuid.uuid4())

    # Determine framework and queue
    worker_type, queue_name = determine_framework_and_queue(
        request.model_type, request.requires_gpu, request.framework_override
    )

    # Ensure worker is available for this job type
    worker_available = await ensure_worker_available(worker_type)
    if not worker_available:
        raise HTTPException(
            status_code=503,
            detail=f"No workers available for {worker_type}. Please try again later.",
        )

    # Create job status
    metadata = {
        "model_type": request.model_type,
        "data_path": request.data_path,
        "hyperparameters": request.hyperparameters,
        "description": request.description,
        "requires_gpu": request.requires_gpu,
        "framework_override": request.framework_override,
    }
    job_status = await create_job_status(
        job_id,
        "training",
        metadata,
        framework=worker_type.split("-")[0],
        worker_type=worker_type,
    )

    # Add to framework-specific queue
    redis_client = await get_redis()
    job_data = {
        "job_id": job_id,
        "model_type": request.model_type,
        "data_path": request.data_path,
        "hyperparameters": request.hyperparameters,
        "description": request.description,
        "requires_gpu": request.requires_gpu,
        "worker_type": worker_type,
        "framework": worker_type.split("-")[0],
    }

    await redis_client.lpush(queue_name, json.dumps(job_data))

    return job_status


@app.post("/jobs/inference", response_model=JobStatus)
async def submit_inference_job(request: InferenceJobRequest):
    """Submit a new inference job to the queue."""
    job_id = str(uuid.uuid4())

    # For inference, we need to determine the framework from the model
    # This would typically be stored in the model metadata
    # For now, we'll use a default queue
    queue_name = INFERENCE_QUEUE

    # Create job status
    metadata = {
        "model_id": request.model_id,
        "input_data": request.input_data,
        "parameters": request.parameters,
    }
    job_status = await create_job_status(job_id, "inference", metadata)

    # Add to inference queue
    redis_client = await get_redis()
    job_data = {
        "job_id": job_id,
        "model_id": request.model_id,
        "input_data": request.input_data,
        "parameters": request.parameters,
    }

    await redis_client.lpush(queue_name, json.dumps(job_data))

    return job_status


@app.get("/jobs/{job_id}/status", response_model=JobStatus)
async def get_job_status_endpoint(job_id: str):
    """Get the status of a specific job."""
    job_status = await get_job_status(job_id)
    if not job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_status


@app.get("/jobs", response_model=List[JobStatus])
async def list_jobs(limit: int = 50):
    """List recent jobs."""
    redis_client = await get_redis()

    # Get all job status keys
    pattern = f"{JOB_STATUS_PREFIX}*"
    keys = await redis_client.keys(pattern)

    jobs = []
    for key in keys[:limit]:
        status_data = await redis_client.get(key)
        if status_data:
            job_status = JobStatus.model_validate_json(status_data)
            jobs.append(job_status)

    # Sort by creation time (newest first)
    jobs.sort(key=lambda x: x.created_at, reverse=True)
    return jobs


@app.get("/frameworks")
async def list_available_frameworks():
    """List available frameworks and their configurations."""
    return {
        "frameworks": list(FRAMEWORK_QUEUES.keys()),
        "model_registry": MODEL_FRAMEWORK_REGISTRY,
        "queues": FRAMEWORK_QUEUES,
    }


@app.get("/workers/status")
async def get_worker_status():
    """Get status of all workers from the worker manager."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{WORKER_MANAGER_URL}/workers/status")
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=503,
                    detail=f"Worker manager unavailable: {response.text}",
                )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=503, detail="Timeout connecting to worker manager"
        )
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Error connecting to worker manager: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        redis_client = await get_redis()
        await redis_client.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
