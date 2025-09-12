#!/usr/bin/env python3
"""
Test script to verify inference job processing is working.
This script submits a training job, waits for completion, then submits an inference job.
"""

import requests
import time
import sys
import json


def test_inference_workflow():
    """Test the complete training -> inference workflow."""
    backend_url = "http://localhost:8000"

    print("🧪 Testing Inference Job Processing...")
    print(f"Backend URL: {backend_url}")
    print()

    # Step 1: Submit a training job first
    print("1. Submitting a training job to create a model...")
    training_job = {
        "model_type": "bert",
        "data_path": "/data/test_training.csv",
        "hyperparameters": {
            "epochs": 2,  # Quick training
            "learning_rate": 0.001,
            "batch_size": 16,
        },
        "description": "Test training for inference",
        "requires_gpu": False,
    }

    try:
        response = requests.post(f"{backend_url}/jobs/training", json=training_job)
        if response.status_code == 200:
            job_data = response.json()
            training_job_id = job_data["job_id"]
            print(f"   ✅ Training job submitted: {training_job_id}")
        else:
            print(f"   ❌ Failed to submit training job: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Failed to submit training job: {e}")
        return False

    # Step 2: Wait for training to complete
    print("2. Waiting for training to complete...")
    max_wait = 60  # 1 minute timeout
    wait_time = 0

    while wait_time < max_wait:
        try:
            response = requests.get(f"{backend_url}/jobs/{training_job_id}/status")
            if response.status_code == 200:
                status_data = response.json()
                status = status_data["status"]
                print(f"   📊 Training status: {status}")

                if status == "completed":
                    model_id = status_data.get("result", {}).get("model_id")
                    if model_id:
                        print(f"   ✅ Training completed! Model ID: {model_id}")
                        break
                    else:
                        print("   ❌ Training completed but no model ID found")
                        return False
                elif status == "failed":
                    print(
                        f"   ❌ Training failed: {status_data.get('error', 'Unknown error')}"
                    )
                    return False
                elif status == "cancelled":
                    print("   ❌ Training was cancelled")
                    return False
            else:
                print(f"   ❌ Failed to get training status: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Error checking training status: {e}")
            return False

        time.sleep(2)
        wait_time += 2

    if wait_time >= max_wait:
        print("   ❌ Training timed out")
        return False

    # Step 3: Submit an inference job
    print("3. Submitting an inference job...")
    inference_job = {
        "model_id": model_id,
        "input_data": "This is a great product! I love it!",
        "parameters": {"temperature": 0.7, "max_length": 100},
    }

    try:
        response = requests.post(f"{backend_url}/jobs/inference", json=inference_job)
        if response.status_code == 200:
            job_data = response.json()
            inference_job_id = job_data["job_id"]
            print(f"   ✅ Inference job submitted: {inference_job_id}")
        else:
            print(f"   ❌ Failed to submit inference job: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Failed to submit inference job: {e}")
        return False

    # Step 4: Monitor inference job progress
    print("4. Monitoring inference job progress...")
    max_wait = 30  # 30 second timeout for inference
    wait_time = 0

    while wait_time < max_wait:
        try:
            response = requests.get(f"{backend_url}/jobs/{inference_job_id}/status")
            if response.status_code == 200:
                status_data = response.json()
                status = status_data["status"]
                progress = status_data.get("result", {}).get("progress", 0)
                stage = status_data.get("result", {}).get("stage", "")

                print(
                    f"   📊 Inference status: {status} (progress: {progress}, stage: {stage})"
                )

                if status == "completed":
                    result = status_data.get("result", {})
                    prediction = result.get("prediction", "unknown")
                    confidence = result.get("confidence", 0)
                    print(f"   ✅ Inference completed!")
                    print(f"   📊 Prediction: {prediction}")
                    print(f"   📊 Confidence: {confidence}")
                    return True
                elif status == "failed":
                    print(
                        f"   ❌ Inference failed: {status_data.get('error', 'Unknown error')}"
                    )
                    return False
                elif status == "cancelled":
                    print("   ❌ Inference was cancelled")
                    return False
            else:
                print(f"   ❌ Failed to get inference status: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Error checking inference status: {e}")
            return False

        time.sleep(1)
        wait_time += 1

    print("   ❌ Inference timed out")
    return False


if __name__ == "__main__":
    print("AI Job Queue System - Inference Fix Test")
    print("=" * 50)
    print()

    # Wait a moment for services to start
    print("⏳ Waiting for services to start...")
    time.sleep(3)

    success = test_inference_workflow()

    if not success:
        print()
        print("❌ Inference test failed!")
        print("💡 Make sure the backend and workers are running:")
        print("   - Backend: ./scripts/run_local_dynamic.sh")
        print("   - Or full stack: ./scripts/run_full_stack.sh")
        sys.exit(1)
    else:
        print()
        print("✅ Inference jobs are now working correctly!")
        print("🎉 You can now submit inference jobs through the web interface!")
        sys.exit(0)
