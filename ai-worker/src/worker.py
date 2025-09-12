import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Optional

import redis.asyncio as redis

try:
    from .config import Config
    from .gpu_manager import GPUMemoryManager, JobQueueManager
except ImportError:
    # Fallback for direct execution
    from config import Config
    from gpu_manager import GPUMemoryManager, JobQueueManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Framework-specific imports
def import_framework_libraries():
    """Import libraries based on the worker's framework."""
    framework = os.getenv("MODEL_FRAMEWORK", "pytorch")
    worker_type = os.getenv("WORKER_TYPE", "pytorch-2.1")

    if framework == "pytorch":
        try:
            import torch
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                TrainingArguments,
                Trainer,
            )
            from datasets import Dataset

            logger.info(f"PyTorch {torch.__version__} loaded successfully")
            return "pytorch", torch
        except ImportError as e:
            logger.error(f"Failed to import PyTorch libraries: {e}")
            return None, None

    elif framework == "tensorflow":
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            import tensorflow_datasets as tfds

            logger.info(f"TensorFlow {tf.__version__} loaded successfully")
            return "tensorflow", tf
        except ImportError as e:
            logger.error(f"Failed to import TensorFlow libraries: {e}")
            return None, None

    elif framework == "sklearn":
        try:
            import sklearn
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC
            from sklearn.linear_model import LogisticRegression

            logger.info(f"Scikit-learn {sklearn.__version__} loaded successfully")
            return "sklearn", sklearn
        except ImportError as e:
            logger.error(f"Failed to import scikit-learn libraries: {e}")
            return None, None

    else:
        logger.error(f"Unknown framework: {framework}")
        return None, None


# Import framework libraries
FRAMEWORK, FRAMEWORK_LIB = import_framework_libraries()


# Queue names - framework-specific
def get_queue_names():
    """Get queue names based on worker type."""
    worker_type = os.getenv("WORKER_TYPE", "pytorch-2.1")

    if worker_type == "pytorch-2.0":
        return "ai:training:pytorch-2.0:queue", "ai:inference:pytorch-2.0:queue"
    elif worker_type == "pytorch-2.1":
        return "ai:training:pytorch-2.1:queue", "ai:inference:pytorch-2.1:queue"
    elif worker_type == "pytorch-2.0-gpu":
        return "ai:training:pytorch-2.0-gpu:queue", "ai:inference:pytorch-2.0-gpu:queue"
    elif worker_type == "pytorch-2.1-gpu":
        return "ai:training:pytorch-2.1-gpu:queue", "ai:inference:pytorch-2.1-gpu:queue"
    elif worker_type == "tensorflow":
        return "ai:training:tensorflow:queue", "ai:inference:tensorflow:queue"
    elif worker_type == "sklearn":
        return "ai:training:sklearn:queue", "ai:inference:sklearn:queue"
    else:
        # Fallback to default queues
        return "ai:training:queue", "ai:inference:queue"


TRAINING_QUEUE, INFERENCE_QUEUE = get_queue_names()
JOB_STATUS_PREFIX = "ai:job:status:"


class AIWorker:
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.running = True
        self.models = {}  # Cache for loaded models
        self.active_jobs = {}  # Track active jobs
        self.semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_JOBS)

        # Framework information
        self.framework = FRAMEWORK
        self.worker_type = os.getenv("WORKER_TYPE", "unknown")
        self.framework_lib = FRAMEWORK_LIB

        if not self.framework or not self.framework_lib:
            logger.error("Failed to initialize framework libraries. Exiting.")
            sys.exit(1)

        # Initialize GPU memory manager if GPU is enabled
        if Config.USE_GPU and "gpu" in self.worker_type:
            self.gpu_manager = GPUMemoryManager(safety_margin_mb=1000)
            self.job_queue_manager = JobQueueManager(self.gpu_manager)
        else:
            self.gpu_manager = None
            self.job_queue_manager = None

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

    async def check_job_cancellation(self, job_id: str) -> bool:
        """Check if a job has been cancelled."""
        status_key = f"{JOB_STATUS_PREFIX}{job_id}"
        status_data = await self.redis_client.get(status_key)

        if status_data:
            job_status = json.loads(status_data)
            return job_status.get("status") == "cancelled"

        return False

    async def process_cancellations(self):
        """Process job cancellation requests."""
        try:
            # Check for cancellation requests
            cancellation_data = await self.redis_client.blpop(
                "ai:job:cancellations", timeout=0.1
            )
            if cancellation_data:
                cancellation = json.loads(cancellation_data[1])
                job_id = cancellation["job_id"]
                logger.info(f"Processing cancellation for job {job_id}")

                # Check if job is currently running and stop it
                status_key = f"{JOB_STATUS_PREFIX}{job_id}"
                status_data = await self.redis_client.get(status_key)

                if status_data:
                    job_status = json.loads(status_data)
                    if job_status.get("status") == "running":
                        # Mark job as cancelled
                        await self.update_job_status(
                            job_id, "cancelled", error="Job cancelled by user"
                        )
                        logger.info(f"Job {job_id} marked as cancelled")

        except Exception as e:
            logger.error(f"Error processing cancellations: {e}")

    async def process_training_job_with_semaphore(self, job_data: Dict):
        """Process a training job with resource limits."""
        async with self.semaphore:
            await self.process_training_job(job_data)

    async def process_inference_job_with_semaphore(self, job_data: Dict):
        """Process an inference job with resource limits."""
        async with self.semaphore:
            await self.process_inference_job(job_data)

    async def process_training_job(self, job_data: Dict):
        """Process a training job using the appropriate framework."""
        job_id = job_data["job_id"]
        model_type = job_data["model_type"]
        data_path = job_data["data_path"]
        hyperparameters = job_data.get("hyperparameters", {})
        requires_gpu = job_data.get("requires_gpu", Config.USE_GPU)

        logger.info(
            f"Starting training job {job_id} for model type: {model_type} using {self.framework}"
        )

        # Check if job was cancelled before starting
        if await self.check_job_cancellation(job_id):
            logger.info(f"Job {job_id} was cancelled before processing")
            return

        try:
            # GPU memory check and optimization
            if requires_gpu and self.gpu_manager:
                # Check if we can start the job
                can_start, reason = await self.job_queue_manager.can_start_job(job_data)
                if not can_start:
                    logger.warning(f"Job {job_id} cannot start: {reason}")

                    # Wait for GPU memory to become available
                    batch_size = hyperparameters.get("batch_size", 1)
                    required_memory = await self.gpu_manager.estimate_model_memory(
                        model_type, batch_size
                    )

                    memory_available = await self.gpu_manager.wait_for_memory(
                        required_memory,
                        device_id=job_data.get("gpu_device", 0),
                        timeout=300,
                    )

                    if not memory_available:
                        error_msg = f"GPU memory timeout. Required: {required_memory}MB"
                        await self.update_job_status(job_id, "failed", error=error_msg)
                        return

                    # Optimize job parameters based on available memory
                    job_data = await self.gpu_manager.optimize_job_parameters(job_data)
                    hyperparameters = job_data.get("hyperparameters", {})
                    logger.info(f"Job {job_id} parameters optimized for GPU memory")

            # Allocate GPU memory if needed
            if requires_gpu and self.gpu_manager:
                batch_size = hyperparameters.get("batch_size", 8)
                required_memory = await self.gpu_manager.estimate_model_memory(
                    model_type, batch_size
                )

                memory_allocated = await self.gpu_manager.allocate_memory(
                    job_id, required_memory, device_id=job_data.get("gpu_device", 0)
                )

                if not memory_allocated:
                    error_msg = f"Failed to allocate {required_memory}MB GPU memory"
                    await self.update_job_status(job_id, "failed", error=error_msg)
                    return

            # Update status to running
            await self.update_job_status(job_id, "running")

            # Framework-specific training
            if self.framework == "pytorch":
                result = await self._train_pytorch_model(
                    job_id, model_type, data_path, hyperparameters
                )
            elif self.framework == "tensorflow":
                result = await self._train_tensorflow_model(
                    job_id, model_type, data_path, hyperparameters
                )
            elif self.framework == "sklearn":
                result = await self._train_sklearn_model(
                    job_id, model_type, data_path, hyperparameters
                )
            else:
                raise ValueError(f"Unsupported framework: {self.framework}")

            # Update status to completed
            await self.update_job_status(job_id, "completed", result)

            logger.info(f"Training job {job_id} completed successfully")

            # Cleanup GPU memory after successful completion
            if requires_gpu and self.gpu_manager:
                await self.gpu_manager.deallocate_memory(job_id)

        except Exception as e:
            logger.error(f"Training job {job_id} failed: {str(e)}")

            # Cleanup GPU memory even on failure
            if requires_gpu and self.gpu_manager:
                await self.gpu_manager.deallocate_memory(job_id)

            await self.update_job_status(job_id, "failed", error=str(e))

    async def _train_pytorch_model(
        self, job_id: str, model_type: str, data_path: str, hyperparameters: Dict
    ) -> Dict:
        """Train a PyTorch model."""
        import torch
        import numpy as np

        # Simulate PyTorch training
        epochs = hyperparameters.get("epochs", 3)
        batch_size = hyperparameters.get("batch_size", 8)

        for epoch in range(epochs):
            # Check for cancellation before each epoch
            if await self.check_job_cancellation(job_id):
                logger.info(
                    f"Job {job_id} cancelled during training at epoch {epoch + 1}"
                )
                return {"cancelled": True, "epoch": epoch + 1}

            logger.info(f"PyTorch training epoch {epoch + 1}/{epochs}")
            # Simulate training time
            await asyncio.sleep(2)

            # Update progress
            progress = (epoch + 1) / epochs
            await self.update_job_status(
                job_id,
                "running",
                {"progress": progress, "epoch": epoch + 1, "framework": "pytorch"},
            )

        # Simulate model saving
        model_id = f"pytorch_model_{job_id}"
        model_path = os.path.join(Config.MODEL_STORAGE_PATH, model_id)
        os.makedirs(model_path, exist_ok=True)

        # Save model info
        model_info = {
            "model_id": model_id,
            "model_type": model_type,
            "framework": "pytorch",
            "pytorch_version": torch.__version__,
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

        return {
            "model_id": model_id,
            "model_path": model_path,
            "framework": "pytorch",
            "metrics": model_info["metrics"],
        }

    async def _train_tensorflow_model(
        self, job_id: str, model_type: str, data_path: str, hyperparameters: Dict
    ) -> Dict:
        """Train a TensorFlow model."""
        import tensorflow as tf

        # Simulate TensorFlow training
        epochs = hyperparameters.get("epochs", 3)
        batch_size = hyperparameters.get("batch_size", 32)

        for epoch in range(epochs):
            logger.info(f"TensorFlow training epoch {epoch + 1}/{epochs}")
            # Simulate training time
            await asyncio.sleep(2)

            # Update progress
            progress = (epoch + 1) / epochs
            await self.update_job_status(
                job_id,
                "running",
                {"progress": progress, "epoch": epoch + 1, "framework": "tensorflow"},
            )

        # Simulate model saving
        model_id = f"tensorflow_model_{job_id}"
        model_path = os.path.join(Config.MODEL_STORAGE_PATH, model_id)
        os.makedirs(model_path, exist_ok=True)

        # Save model info
        model_info = {
            "model_id": model_id,
            "model_type": model_type,
            "framework": "tensorflow",
            "tensorflow_version": tf.__version__,
            "training_data": data_path,
            "hyperparameters": hyperparameters,
            "created_at": datetime.utcnow().isoformat(),
            "metrics": {
                "accuracy": 0.92,
                "loss": 0.08,
                "training_time": epochs * 2,
            },
        }

        with open(os.path.join(model_path, "model_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)

        return {
            "model_id": model_id,
            "model_path": model_path,
            "framework": "tensorflow",
            "metrics": model_info["metrics"],
        }

    async def _train_sklearn_model(
        self, job_id: str, model_type: str, data_path: str, hyperparameters: Dict
    ) -> Dict:
        """Train a scikit-learn model."""
        import sklearn

        # Simulate scikit-learn training
        logger.info(f"Training scikit-learn {model_type} model")
        # Simulate training time (usually faster than deep learning)
        await asyncio.sleep(1)

        # Update progress
        await self.update_job_status(
            job_id, "running", {"progress": 1.0, "framework": "sklearn"}
        )

        return {
            "model_id": f"sklearn_model_{job_id}",
            "model_path": os.path.join(
                Config.MODEL_STORAGE_PATH, f"sklearn_model_{job_id}"
            ),
            "framework": "sklearn",
            "metrics": {
                "accuracy": 0.90,
                "training_time": 1,
            },
        }

    async def process_inference_job(self, job_data: Dict):
        """Process an inference job using the appropriate framework."""
        job_id = job_data["job_id"]
        model_id = job_data["model_id"]
        input_data = job_data["input_data"]
        parameters = job_data.get("parameters", {})
        requires_gpu = job_data.get("requires_gpu", Config.USE_GPU)
        model_type = job_data.get("model_type", "unknown")

        logger.info(
            f"Starting inference job {job_id} for model type: {model_type} using {self.framework}"
        )

        try:
            # GPU memory check and optimization
            if requires_gpu and self.gpu_manager:
                # Check if we can start the job
                can_start, reason = await self.job_queue_manager.can_start_job(job_data)
                if not can_start:
                    logger.warning(f"Job {job_id} cannot start: {reason}")

                    # Wait for GPU memory to become available
                    required_memory = await self.gpu_manager.estimate_model_memory(
                        model_type, 1
                    )

                    memory_available = await self.gpu_manager.wait_for_memory(
                        required_memory,
                        device_id=job_data.get("gpu_device", 0),
                        timeout=300,
                    )

                    if not memory_available:
                        error_msg = f"GPU memory timeout. Required: {required_memory}MB"
                        await self.update_job_status(job_id, "failed", error=error_msg)
                        return

                    # Optimize job parameters based on available memory
                    job_data = await self.gpu_manager.optimize_job_parameters(job_data)
                    parameters = job_data.get("parameters", {})
                    logger.info(f"Job {job_id} parameters optimized for GPU memory")

            # Allocate GPU memory if needed
            if requires_gpu and self.gpu_manager:
                required_memory = await self.gpu_manager.estimate_model_memory(
                    model_type, 1
                )

                memory_allocated = await self.gpu_manager.allocate_memory(
                    job_id, required_memory, device_id=job_data.get("gpu_device", 0)
                )

                if not memory_allocated:
                    error_msg = f"Failed to allocate {required_memory}MB GPU memory"
                    await self.update_job_status(job_id, "failed", error=error_msg)
                    return

            # Update status to running
            await self.update_job_status(job_id, "running")

            # Framework-specific inference
            if self.framework == "pytorch":
                result = await self._infer_pytorch_model(
                    job_id, model_type, input_data, parameters
                )
            elif self.framework == "tensorflow":
                result = await self._infer_tensorflow_model(
                    job_id, model_type, input_data, parameters
                )
            elif self.framework == "sklearn":
                result = await self._infer_sklearn_model(
                    job_id, model_type, input_data, parameters
                )
            else:
                raise ValueError(f"Unsupported framework: {self.framework}")

            # Update status to completed
            await self.update_job_status(job_id, "completed", result)

            logger.info(f"Inference job {job_id} completed successfully")

            # Cleanup GPU memory after successful completion
            if requires_gpu and self.gpu_manager:
                await self.gpu_manager.deallocate_memory(job_id)

        except Exception as e:
            logger.error(f"Inference job {job_id} failed: {str(e)}")

            # Cleanup GPU memory even on failure
            if requires_gpu and self.gpu_manager:
                await self.gpu_manager.deallocate_memory(job_id)

            await self.update_job_status(job_id, "failed", error=str(e))

    async def _infer_pytorch_model(
        self, job_id: str, model_type: str, input_data: str, parameters: Dict
    ) -> Dict:
        """Run inference with a PyTorch model."""
        import torch
        import numpy as np

        # Simulate PyTorch inference
        logger.info(f"Running inference with PyTorch {model_type} model")
        # Simulate inference time
        await asyncio.sleep(1)

        # Update progress
        await self.update_job_status(
            job_id, "running", {"progress": 1.0, "framework": "pytorch"}
        )

        return {
            "model_id": f"pytorch_model_{job_id}",
            "model_path": os.path.join(
                Config.MODEL_STORAGE_PATH, f"pytorch_model_{job_id}"
            ),
            "framework": "pytorch",
            "metrics": {
                "accuracy": 0.95,
                "inference_time": 1,
            },
        }

    async def _infer_tensorflow_model(
        self, job_id: str, model_type: str, input_data: str, parameters: Dict
    ) -> Dict:
        """Run inference with a TensorFlow model."""
        import tensorflow as tf
        import numpy as np

        # Simulate TensorFlow inference
        logger.info(f"Running inference with TensorFlow {model_type} model")
        # Simulate inference time
        await asyncio.sleep(1)

        # Update progress
        await self.update_job_status(
            job_id, "running", {"progress": 1.0, "framework": "tensorflow"}
        )

        return {
            "model_id": f"tensorflow_model_{job_id}",
            "model_path": os.path.join(
                Config.MODEL_STORAGE_PATH, f"tensorflow_model_{job_id}"
            ),
            "framework": "tensorflow",
            "metrics": {
                "accuracy": 0.92,
                "inference_time": 1,
            },
        }

    async def _infer_sklearn_model(
        self, job_id: str, model_type: str, input_data: str, parameters: Dict
    ) -> Dict:
        """Run inference with a scikit-learn model."""
        import sklearn
        import numpy as np

        # Simulate scikit-learn inference
        logger.info(f"Running inference with scikit-learn {model_type} model")
        # Simulate inference time
        await asyncio.sleep(1)

        # Update progress
        await self.update_job_status(
            job_id, "running", {"progress": 1.0, "framework": "sklearn"}
        )

        return {
            "model_id": f"sklearn_model_{job_id}",
            "model_path": os.path.join(
                Config.MODEL_STORAGE_PATH, f"sklearn_model_{job_id}"
            ),
            "framework": "sklearn",
            "metrics": {
                "accuracy": 0.90,
                "inference_time": 1,
            },
        }

    async def run(self):
        """Main worker loop to process jobs from Redis queues."""
        training_queue, inference_queue = get_queue_names()

        logger.info(f"Starting worker loop for {self.worker_type}")
        logger.info(f"Training queue: {training_queue}")
        logger.info(f"Inference queue: {inference_queue}")

        while self.running:
            try:
                # Process cancellation requests
                await self.process_cancellations()

                # Check for training jobs
                training_job = await self.redis_client.blpop(training_queue, timeout=1)
                if training_job:
                    job_data = json.loads(training_job[1])
                    logger.info(f"Received training job: {job_data['job_id']}")
                    asyncio.create_task(
                        self.process_training_job_with_semaphore(job_data)
                    )

                # Check for inference jobs
                inference_job = await self.redis_client.blpop(
                    inference_queue, timeout=1
                )
                if inference_job:
                    job_data = json.loads(inference_job[1])
                    logger.info(f"Received inference job: {job_data['job_id']}")
                    asyncio.create_task(
                        self.process_inference_job_with_semaphore(job_data)
                    )

            except asyncio.CancelledError:
                logger.info("Worker loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying

        logger.info("Worker loop stopped")

    def stop(self):
        """Stop the worker."""
        self.running = False


async def main():
    """Main function to run the AI worker."""
    worker = AIWorker()

    try:
        # Connect to Redis
        await worker.connect_redis()

        # Get queue names
        training_queue, inference_queue = get_queue_names()
        logger.info(f"Worker started for {worker.worker_type}")
        logger.info(f"Listening on queues: {training_queue}, {inference_queue}")

        # Start the worker loop
        await worker.run()

    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.error(f"Worker error: {e}")
        raise
    finally:
        if worker.redis_client:
            await worker.redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())
