import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# Initialize FastAPI app
app = FastAPI(
    title="AI Job Queue API",
    description="API for submitting and monitoring AI training and inference jobs",
    version="1.0.0",
)

# Redis connection
redis_client: Optional[redis.Redis] = None


# Pydantic models
class TrainingJobRequest(BaseModel):
    model_type: str = Field(..., description="Type of model to train")
    data_path: str = Field(..., description="Path to training data")
    hyperparameters: Optional[Dict] = Field(
        default_factory=dict, description="Training hyperparameters"
    )
    description: Optional[str] = Field(None, description="Job description")


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


# Redis keys
TRAINING_QUEUE = "ai:training:queue"
INFERENCE_QUEUE = "ai:inference:queue"
JOB_STATUS_PREFIX = "ai:job:status:"


@app.on_event("startup")
async def startup_event():
    """Initialize Redis connection on startup."""
    global redis_client
    redis_client = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)
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


async def create_job_status(job_id: str, job_type: str, metadata: Dict) -> JobStatus:
    """Create a new job status entry."""
    now = datetime.utcnow()
    job_status = JobStatus(
        job_id=job_id,
        job_type=job_type,
        status="pending",
        created_at=now,
        updated_at=now,
        metadata=metadata,
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
    """Submit a new training job to the queue."""
    job_id = str(uuid.uuid4())

    # Create job status
    metadata = {
        "model_type": request.model_type,
        "data_path": request.data_path,
        "hyperparameters": request.hyperparameters,
        "description": request.description,
    }
    job_status = await create_job_status(job_id, "training", metadata)

    # Add to training queue
    redis_client = await get_redis()
    job_data = {
        "job_id": job_id,
        "model_type": request.model_type,
        "data_path": request.data_path,
        "hyperparameters": request.hyperparameters,
        "description": request.description,
    }

    await redis_client.lpush(TRAINING_QUEUE, json.dumps(job_data))

    return job_status


@app.post("/jobs/inference", response_model=JobStatus)
async def submit_inference_job(request: InferenceJobRequest):
    """Submit a new inference job to the queue."""
    job_id = str(uuid.uuid4())

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

    await redis_client.lpush(INFERENCE_QUEUE, json.dumps(job_data))

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
