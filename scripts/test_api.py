#!/usr/bin/env python3
"""
Test script for the AI Job Queue API
"""

import asyncio
import json
import time
from typing import Dict

import httpx

BASE_URL = "http://localhost:8000"


async def test_health():
    """Test the health endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")


async def submit_training_job() -> str:
    """Submit a training job and return the job ID."""
    training_data = {
        "model_type": "bert",  # Use registered model type for PyTorch 2.1
        "data_path": "/data/training_data.csv",
        "hyperparameters": {"epochs": 5, "learning_rate": 0.001, "batch_size": 32},
        "description": "Test training job for sentiment analysis",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/jobs/training", json=training_data)

        if response.status_code == 200:
            job = response.json()
            print(f"Training job submitted: {job['job_id']}")
            return job["job_id"]
        else:
            print(f"Failed to submit training job: {response.status_code}")
            return None


async def submit_inference_job(model_id: str) -> str:
    """Submit an inference job and return the job ID."""
    inference_data = {
        "model_id": model_id,
        "input_data": "This is a great product!",
        "parameters": {"temperature": 0.7, "max_length": 100},
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/jobs/inference", json=inference_data)

        if response.status_code == 200:
            job = response.json()
            print(f"Inference job submitted: {job['job_id']}")
            return job["job_id"]
        else:
            print(f"Failed to submit inference job: {response.status_code}")
            return None


async def check_job_status(job_id: str) -> Dict:
    """Check the status of a job."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/jobs/{job_id}/status")

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to get job status: {response.status_code}")
            return None


async def list_jobs():
    """List all jobs."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/jobs")

        if response.status_code == 200:
            jobs = response.json()
            print(f"Found {len(jobs)} jobs:")
            for job in jobs:
                print(f"  - {job['job_id']}: {job['job_type']} ({job['status']})")
        else:
            print(f"Failed to list jobs: {response.status_code}")


async def monitor_job(job_id: str, max_wait: int = 60):
    """Monitor a job until completion or timeout."""
    print(f"Monitoring job {job_id}...")

    start_time = time.time()
    while time.time() - start_time < max_wait:
        status = await check_job_status(job_id)
        if status:
            print(f"Status: {status['status']}")
            if status["status"] in ["completed", "failed"]:
                if status["status"] == "completed":
                    print(f"Job completed successfully!")
                    if "result" in status:
                        print(f"Result: {json.dumps(status['result'], indent=2)}")
                else:
                    print(f"Job failed: {status.get('error', 'Unknown error')}")
                return status
            elif status["status"] == "running" and "result" in status:
                # Show progress for running jobs
                result = status["result"]
                if "progress" in result:
                    print(f"Progress: {result['progress']:.1%}")
                if "epoch" in result:
                    print(f"Epoch: {result['epoch']}")

        await asyncio.sleep(2)

    print(f"Job monitoring timed out after {max_wait} seconds")
    return None


async def submit_multiple_framework_jobs():
    """Submit training jobs for different frameworks."""
    frameworks = [
        {"model_type": "bert", "description": "PyTorch 2.1 BERT model"},
        {"model_type": "resnet", "description": "PyTorch 2.0 ResNet model"},
        {"model_type": "inception", "description": "TensorFlow Inception model"},
        {"model_type": "random_forest", "description": "Scikit-learn Random Forest"},
    ]

    job_ids = []
    for framework in frameworks:
        training_data = {
            "model_type": framework["model_type"],
            "data_path": "/data/training_data.csv",
            "hyperparameters": {"epochs": 3, "learning_rate": 0.001, "batch_size": 16},
            "description": framework["description"],
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URL}/jobs/training", json=training_data
            )
            if response.status_code == 200:
                job = response.json()
                print(f"✅ {framework['description']} job submitted: {job['job_id']}")
                job_ids.append(job["job_id"])
            else:
                print(
                    f"❌ Failed to submit {framework['description']}: {response.status_code}"
                )

    return job_ids


async def main():
    """Main test function."""
    print("=== AI Job Queue API Test ===\n")

    # Test health endpoint
    print("1. Testing health endpoint...")
    await test_health()
    print()

    # Submit single training job
    print("2. Submitting single training job...")
    training_job_id = await submit_training_job()
    if not training_job_id:
        return
    print()

    # Monitor training job
    print("3. Monitoring training job...")
    training_result = await monitor_job(training_job_id, max_wait=30)
    print()

    if training_result and training_result["status"] == "completed":
        # Extract model ID from training result
        model_id = training_result["result"]["model_id"]

        # Submit inference job
        print("4. Submitting inference job...")
        inference_job_id = await submit_inference_job(model_id)
        if inference_job_id:
            print()

            # Monitor inference job
            print("5. Monitoring inference job...")
            await monitor_job(inference_job_id, max_wait=20)
            print()

    # Test multiple frameworks
    print("6. Testing multiple frameworks...")
    multi_job_ids = await submit_multiple_framework_jobs()
    print()

    # List all jobs
    print("7. Listing all jobs...")
    await list_jobs()


if __name__ == "__main__":
    asyncio.run(main())
