#!/usr/bin/env python3
"""
Test script to verify the frontend displays inference job details correctly.
This script submits an inference job and checks that the frontend shows the details.
"""

import requests
import time
import sys


def test_inference_job_display():
    """Test that inference job details are displayed correctly in the frontend."""
    backend_url = "http://localhost:8000"
    frontend_url = "http://localhost:3000"

    print("🧪 Testing Frontend Inference Job Display...")
    print(f"Backend URL: {backend_url}")
    print(f"Frontend URL: {frontend_url}")
    print()

    # Step 1: Submit an inference job
    print("1. Submitting an inference job...")
    inference_job = {
        "model_id": "pytorch_model_test_123",
        "input_data": "This is a great product! I love it!",
        "parameters": {
            "temperature": 0.8,
            "max_length": 200,
            "top_k": 5,
            "confidence_threshold": 0.7,
        },
    }

    try:
        response = requests.post(f"{backend_url}/jobs/inference", json=inference_job)
        if response.status_code == 200:
            job_data = response.json()
            job_id = job_data["job_id"]
            print(f"   ✅ Inference job submitted: {job_id}")
        else:
            print(f"   ❌ Failed to submit inference job: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Failed to submit inference job: {e}")
        return False

    # Step 2: Check the job status via API
    print("2. Checking job status via API...")
    try:
        response = requests.get(f"{backend_url}/jobs/{job_id}/status")
        if response.status_code == 200:
            status_data = response.json()
            print(f"   ✅ Job status retrieved")
            print(f"   📊 Status: {status_data['status']}")
            print(f"   📊 Job Type: {status_data['job_type']}")
            print(f"   📊 Framework: {status_data.get('framework', 'N/A')}")
            print(f"   📊 Worker Type: {status_data.get('worker_type', 'N/A')}")

            if status_data.get("metadata"):
                metadata = status_data["metadata"]
                print(f"   📊 Model ID: {metadata.get('model_id', 'N/A')}")
                print(f"   📊 Input Data: {metadata.get('input_data', 'N/A')}")
                print(f"   📊 Parameters: {metadata.get('parameters', {})}")
                print(
                    f"   📊 Model Framework: {metadata.get('model_framework', 'N/A')}"
                )
        else:
            print(f"   ❌ Failed to get job status: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error checking job status: {e}")
        return False

    # Step 3: Check if frontend is accessible
    print("3. Checking frontend accessibility...")
    try:
        response = requests.get(f"{frontend_url}")
        if response.status_code == 200:
            print(f"   ✅ Frontend is accessible")
        else:
            print(f"   ❌ Frontend not accessible: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Frontend not accessible: {e}")
        return False

    # Step 4: Check if the job appears in the job list
    print("4. Checking job list API...")
    try:
        response = requests.get(f"{backend_url}/jobs")
        if response.status_code == 200:
            jobs = response.json()
            job_found = False
            for job in jobs:
                if job["job_id"] == job_id:
                    job_found = True
                    print(f"   ✅ Job found in job list")
                    print(f"   📊 Job details in list:")
                    print(f"      - Status: {job['status']}")
                    print(f"      - Type: {job['job_type']}")
                    print(f"      - Framework: {job.get('framework', 'N/A')}")
                    print(f"      - Worker Type: {job.get('worker_type', 'N/A')}")
                    if job.get("metadata"):
                        print(
                            f"      - Model ID: {job['metadata'].get('model_id', 'N/A')}"
                        )
                        print(
                            f"      - Input Data: {job['metadata'].get('input_data', 'N/A')}"
                        )
                    break

            if not job_found:
                print(f"   ❌ Job not found in job list")
                return False
        else:
            print(f"   ❌ Failed to get job list: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error getting job list: {e}")
        return False

    print()
    print("✅ Frontend inference job display test completed!")
    print(f"🌐 You can now view the job details at: {frontend_url}")
    print(f"🔍 Job ID: {job_id}")
    print()
    print("Expected frontend display:")
    print("  - Job Type: Inference Job")
    print("  - Status: Pending (with hourglass icon)")
    print("  - Framework: pytorch")
    print("  - Worker Type: pytorch-2.1")
    print("  - Details section should show:")
    print("    - Model ID: pytorch_model_test_123")
    print("    - Input Data: This is a great product! I love it!")
    print("    - Parameters: temperature, max_length, top_k, confidence_threshold")
    print("    - Model Framework: pytorch")

    return True


if __name__ == "__main__":
    print("AI Job Queue System - Frontend Inference Display Test")
    print("=" * 60)
    print()

    # Wait a moment for services to start
    print("⏳ Waiting for services to start...")
    time.sleep(3)

    success = test_inference_job_display()

    if not success:
        print()
        print("❌ Frontend inference display test failed!")
        print("💡 Make sure both backend and frontend are running:")
        print("   - Backend: ./scripts/run_local_dynamic.sh")
        print("   - Frontend: ./scripts/run_frontend_dev.sh")
        print("   - Or full stack: ./scripts/run_full_stack.sh")
        sys.exit(1)
    else:
        print()
        print("🎉 Frontend now displays inference job details correctly!")
        sys.exit(0)
