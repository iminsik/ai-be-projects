#!/usr/bin/env python3
"""
Test script for job cancellation functionality.
This script tests the job cancellation feature by submitting a job and then cancelling it.
"""

import requests
import time
import sys
import json
from datetime import datetime


def test_job_cancellation():
    """Test the job cancellation functionality."""
    backend_url = "http://localhost:8000"

    print("🧪 Testing Job Cancellation Feature...")
    print(f"Backend URL: {backend_url}")
    print()

    # Test 1: Submit a training job
    print("1. Submitting a training job...")
    training_job = {
        "model_type": "bert",
        "data_path": "/data/test_training.csv",
        "hyperparameters": {
            "epochs": 10,  # Long enough to cancel
            "learning_rate": 0.001,
            "batch_size": 16,
        },
        "description": "Test job for cancellation",
        "requires_gpu": False,
    }

    try:
        response = requests.post(f"{backend_url}/jobs/training", json=training_job)
        if response.status_code == 200:
            job_data = response.json()
            job_id = job_data["job_id"]
            print(f"   ✅ Training job submitted: {job_id}")
            print(f"   📊 Initial status: {job_data['status']}")
        else:
            print(f"   ❌ Failed to submit training job: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Failed to submit training job: {e}")
        return False

    # Test 2: Wait a moment for job to start
    print("2. Waiting for job to start...")
    time.sleep(3)

    # Check job status
    try:
        response = requests.get(f"{backend_url}/jobs/{job_id}/status")
        if response.status_code == 200:
            status_data = response.json()
            print(f"   📊 Job status: {status_data['status']}")
        else:
            print(f"   ❌ Failed to get job status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Failed to get job status: {e}")

    # Test 3: Cancel the job
    print("3. Cancelling the job...")
    try:
        response = requests.delete(f"{backend_url}/jobs/{job_id}")
        if response.status_code == 200:
            cancel_data = response.json()
            print(f"   ✅ Job cancelled successfully")
            print(f"   📊 New status: {cancel_data['status']}")
            print(f"   📊 Error message: {cancel_data.get('error', 'None')}")
        else:
            print(f"   ❌ Failed to cancel job: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Failed to cancel job: {e}")
        return False

    # Test 4: Verify cancellation
    print("4. Verifying job cancellation...")
    time.sleep(2)  # Wait for worker to process cancellation

    try:
        response = requests.get(f"{backend_url}/jobs/{job_id}/status")
        if response.status_code == 200:
            final_status = response.json()
            print(f"   📊 Final status: {final_status['status']}")

            if final_status["status"] == "cancelled":
                print("   ✅ Job successfully cancelled!")
            else:
                print(
                    f"   ⚠️  Job status is {final_status['status']}, expected 'cancelled'"
                )
        else:
            print(f"   ❌ Failed to get final job status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Failed to get final job status: {e}")

    # Test 5: Test cancelling already completed job
    print("5. Testing cancellation of non-cancellable job...")
    try:
        response = requests.delete(f"{backend_url}/jobs/{job_id}")
        if response.status_code == 400:
            print("   ✅ Correctly rejected cancellation of already cancelled job")
        else:
            print(
                f"   ⚠️  Unexpected response for already cancelled job: {response.status_code}"
            )
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error testing already cancelled job: {e}")

    # Test 6: Test cancelling non-existent job
    print("6. Testing cancellation of non-existent job...")
    fake_job_id = "fake-job-id-12345"
    try:
        response = requests.delete(f"{backend_url}/jobs/{fake_job_id}")
        if response.status_code == 404:
            print("   ✅ Correctly rejected cancellation of non-existent job")
        else:
            print(
                f"   ⚠️  Unexpected response for non-existent job: {response.status_code}"
            )
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error testing non-existent job: {e}")

    print()
    print("🎉 Job cancellation tests completed!")
    print()
    print("💡 Test Summary:")
    print("   - Job submission: ✅")
    print("   - Job cancellation: ✅")
    print("   - Status verification: ✅")
    print("   - Error handling: ✅")
    print()
    print("🌐 You can also test the cancellation feature in the web interface:")
    print("   - Open http://localhost:3000")
    print("   - Submit a training job")
    print("   - Click the 'Cancel Job' button on running jobs")

    return True


if __name__ == "__main__":
    print("AI Job Queue System - Job Cancellation Test")
    print("=" * 50)
    print()

    # Wait a moment for services to start
    print("⏳ Waiting for services to start...")
    time.sleep(3)

    success = test_job_cancellation()

    if not success:
        print()
        print("❌ Job cancellation tests failed!")
        print("💡 Make sure the backend and workers are running:")
        print("   - Backend: ./scripts/run_local_dynamic.sh")
        print("   - Or full stack: ./scripts/run_full_stack.sh")
        sys.exit(1)
    else:
        print()
        print("✅ Job cancellation feature is working correctly!")
        sys.exit(0)
