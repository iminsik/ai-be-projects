import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from typing import Dict, Optional

import redis.asyncio as redis
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import numpy as np
import pandas as pd

try:
    from .config import Config
except ImportError:
    # Fallback for direct execution
    from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Queue names
TRAINING_QUEUE = "ai:training:queue"
INFERENCE_QUEUE = "ai:inference:queue"
JOB_STATUS_PREFIX = "ai:job:status:"


class AIWorker:
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.running = True
        self.models = {}  # Cache for loaded models
        self.active_jobs = {}  # Track active jobs
        self.semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_JOBS)

    async def connect_redis(self):
        """Connect to Redis."""
        self.redis_client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            password=Config.REDIS_PASSWORD,
            decode_responses=True,
        )
        await self.redis_client.ping()
        logger.info("Connected to Redis")

    async def update_job_status(
        self,
        job_id: str,
        status: str,
        result: Optional[Dict] = None,
        error: Optional[str] = None,
    ):
        """Update job status in Redis."""
        status_key = f"{JOB_STATUS_PREFIX}{job_id}"

        status_data = await self.redis_client.get(status_key)
        if status_data:
            job_status = json.loads(status_data)
            job_status["status"] = status
            job_status["updated_at"] = datetime.utcnow().isoformat()
            if result is not None:
                job_status["result"] = result
            if error is not None:
                job_status["error"] = error

            await self.redis_client.set(status_key, json.dumps(job_status), ex=86400)

    async def process_training_job_with_semaphore(self, job_data: Dict):
        """Process a training job with resource limits."""
        async with self.semaphore:
            await self.process_training_job(job_data)

    async def process_inference_job_with_semaphore(self, job_data: Dict):
        """Process an inference job with resource limits."""
        async with self.semaphore:
            await self.process_inference_job(job_data)

    async def process_training_job(self, job_data: Dict):
        """Process a training job."""
        job_id = job_data["job_id"]
        model_type = job_data["model_type"]
        data_path = job_data["data_path"]
        hyperparameters = job_data.get("hyperparameters", {})

        logger.info(f"Starting training job {job_id} for model type: {model_type}")

        try:
            # Update status to running
            await self.update_job_status(job_id, "running")

            # Simulate training process
            # In a real implementation, you would:
            # 1. Load your training data
            # 2. Initialize the model
            # 3. Train the model
            # 4. Save the model

            # For demonstration, we'll simulate training
            epochs = hyperparameters.get("epochs", 3)
            for epoch in range(epochs):
                logger.info(f"Training epoch {epoch + 1}/{epochs}")
                # Simulate training time
                await asyncio.sleep(2)

                # Update progress
                progress = (epoch + 1) / epochs
                await self.update_job_status(
                    job_id, "running", {"progress": progress, "epoch": epoch + 1}
                )

            # Simulate model saving
            model_id = f"model_{job_id}"
            model_path = os.path.join(Config.MODEL_STORAGE_PATH, model_id)
            os.makedirs(model_path, exist_ok=True)

            # Save dummy model info
            model_info = {
                "model_id": model_id,
                "model_type": model_type,
                "training_data": data_path,
                "hyperparameters": hyperparameters,
                "created_at": datetime.utcnow().isoformat(),
                "metrics": {
                    "accuracy": 0.95,
                    "loss": 0.05,
                    "training_time": epochs * 2,
                },
            }

            with open(os.path.join(model_path, "model_info.json"), "w") as f:
                json.dump(model_info, f, indent=2)

            # Update status to completed
            await self.update_job_status(
                job_id,
                "completed",
                {
                    "model_id": model_id,
                    "model_path": model_path,
                    "metrics": model_info["metrics"],
                },
            )

            logger.info(f"Training job {job_id} completed successfully")

        except Exception as e:
            logger.error(f"Training job {job_id} failed: {str(e)}")
            await self.update_job_status(job_id, "failed", error=str(e))

    async def process_inference_job(self, job_data: Dict):
        """Process an inference job."""
        job_id = job_data["job_id"]
        model_id = job_data["model_id"]
        input_data = job_data["input_data"]
        parameters = job_data.get("parameters", {})

        logger.info(f"Starting inference job {job_id} for model: {model_id}")

        try:
            # Update status to running
            await self.update_job_status(job_id, "running")

            # Load model (in cache or from storage)
            if model_id not in self.models:
                model_path = os.path.join(Config.MODEL_STORAGE_PATH, model_id)
                model_info_path = os.path.join(model_path, "model_info.json")

                if not os.path.exists(model_info_path):
                    raise FileNotFoundError(f"Model {model_id} not found")

                # Load model info
                with open(model_info_path, "r") as f:
                    model_info = json.load(f)

                # In a real implementation, you would load the actual model
                # For demonstration, we'll simulate inference
                self.models[model_id] = model_info

            # Simulate inference
            await asyncio.sleep(1)

            # Generate dummy prediction
            prediction = {
                "input": input_data,
                "output": f"Predicted result for: {input_data}",
                "confidence": 0.95,
                "model_id": model_id,
                "inference_time": 1.0,
            }

            # Update status to completed
            await self.update_job_status(
                job_id, "completed", {"prediction": prediction}
            )

            logger.info(f"Inference job {job_id} completed successfully")

        except Exception as e:
            logger.error(f"Inference job {job_id} failed: {str(e)}")
            await self.update_job_status(job_id, "failed", error=str(e))

    async def process_training_queue(self):
        """Process training jobs from the queue."""
        while self.running:
            try:
                # Pop job from training queue (blocking)
                result = await self.redis_client.brpop(TRAINING_QUEUE, timeout=1)
                if result:
                    _, job_data_str = result
                    job_data = json.loads(job_data_str)
                    # Process job concurrently with resource limits
                    asyncio.create_task(
                        self.process_training_job_with_semaphore(job_data)
                    )
            except Exception as e:
                logger.error(f"Error processing training queue: {str(e)}")
                await asyncio.sleep(1)

    async def process_inference_queue(self):
        """Process inference jobs from the queue."""
        while self.running:
            try:
                # Pop job from inference queue (blocking)
                result = await self.redis_client.brpop(INFERENCE_QUEUE, timeout=1)
                if result:
                    _, job_data_str = result
                    job_data = json.loads(job_data_str)
                    # Process job concurrently with resource limits
                    asyncio.create_task(
                        self.process_inference_job_with_semaphore(job_data)
                    )
            except Exception as e:
                logger.error(f"Error processing inference queue: {str(e)}")
                await asyncio.sleep(1)

    async def run(self):
        """Main worker loop."""
        await self.connect_redis()

        # Create model storage directory
        os.makedirs(Config.MODEL_STORAGE_PATH, exist_ok=True)

        logger.info("AI Worker started")
        logger.info(f"Model storage path: {Config.MODEL_STORAGE_PATH}")
        logger.info(f"Max concurrent jobs: {Config.MAX_CONCURRENT_JOBS}")
        logger.info(f"GPU enabled: {Config.USE_GPU}")
        if Config.USE_GPU:
            logger.info(f"GPU device: {Config.GPU_DEVICE}")

        # Run both queues concurrently
        await asyncio.gather(
            self.process_training_queue(), self.process_inference_queue()
        )

    def stop(self):
        """Stop the worker."""
        self.running = False
        logger.info("AI Worker stopping...")

    def get_status(self) -> Dict:
        """Get worker status information."""
        return {
            "worker_id": Config.WORKER_ID,
            "max_concurrent_jobs": Config.MAX_CONCURRENT_JOBS,
            "active_jobs": len(self.active_jobs),
            "available_slots": self.semaphore._value,
            "gpu_enabled": Config.USE_GPU,
            "gpu_device": Config.GPU_DEVICE if Config.USE_GPU else None,
            "models_loaded": len(self.models),
            "running": self.running,
        }


async def main():
    """Main entry point."""
    worker = AIWorker()

    # Handle shutdown signals
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        worker.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await worker.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        worker.stop()
        if worker.redis_client:
            await worker.redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())
