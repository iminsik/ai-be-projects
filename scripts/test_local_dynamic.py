#!/usr/bin/env python3
"""
Test script for local dynamic worker system
"""

import asyncio
import json
import time
import httpx


async def test_local_dynamic_workers():
    """Test the local dynamic worker system."""
    print("🧪 Testing Local Dynamic Worker System...")

    base_url = "http://localhost:8000"
    worker_manager_url = "http://localhost:8001"

    async with httpx.AsyncClient() as client:
        # Test 1: Check if services are running
        print("\n1️⃣ Checking service health...")

        try:
            # Check backend
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                print("✅ Backend API is healthy")
            else:
                print("❌ Backend API is not responding")
                return False
        except Exception as e:
            print(f"❌ Backend API error: {e}")
            return False

        try:
            # Check worker manager
            response = await client.get(f"{worker_manager_url}/workers/status")
            if response.status_code == 200:
                print("✅ Worker Manager is healthy")
                status = response.json()
                print(f"   Active workers: {status['active_workers']}")
            else:
                print("❌ Worker Manager is not responding")
                return False
        except Exception as e:
            print(f"❌ Worker Manager error: {e}")
            return False

        # Test 2: Start a worker
        print("\n2️⃣ Starting a PyTorch worker...")

        try:
            response = await client.post(
                f"{worker_manager_url}/workers/start/pytorch-2.1"
            )
            if response.status_code == 200:
                result = response.json()
                worker_id = result["worker_id"]
                print(f"✅ Worker started: {worker_id}")
            else:
                print(f"❌ Failed to start worker: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Error starting worker: {e}")
            return False

        # Test 3: Check worker status
        print("\n3️⃣ Checking worker status...")

        try:
            response = await client.get(f"{worker_manager_url}/workers/status")
            if response.status_code == 200:
                status = response.json()
                print(f"✅ Total workers: {status['total_workers']}")
                print(f"✅ Active workers: {status['active_workers']}")
                print(
                    f"✅ PyTorch 2.1 workers: {status['workers_by_type']['pytorch-2.1']}"
                )
            else:
                print(f"❌ Failed to get worker status: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Error getting worker status: {e}")
            return False

        # Test 4: Submit a training job
        print("\n4️⃣ Submitting a training job...")

        job_data = {
            "model_type": "bert",
            "data_path": "/data/train.csv",
            "hyperparameters": {"batch_size": 8, "epochs": 3},
            "description": "Test job for local dynamic workers",
            "requires_gpu": False,
        }

        try:
            response = await client.post(f"{base_url}/jobs/training", json=job_data)
            if response.status_code == 200:
                job = response.json()
                print(f"✅ Job submitted: {job['job_id']}")
                print(f"   Status: {job['status']}")
                print(f"   Worker Type: {job['metadata']['worker_type']}")
            else:
                print(f"❌ Failed to submit job: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Error submitting job: {e}")
            return False

        # Test 5: Wait and check job status
        print("\n5️⃣ Checking job status...")

        try:
            # Wait a bit for job processing
            await asyncio.sleep(2)

            response = await client.get(f"{base_url}/jobs/{job['job_id']}/status")
            if response.status_code == 200:
                job_status = response.json()
                print(f"✅ Job status: {job_status['status']}")
                print(f"   Progress: {job_status.get('progress', 'N/A')}")
            else:
                print(f"❌ Failed to get job status: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Error getting job status: {e}")
            return False

        # Test 6: Clean up - stop the worker
        print("\n6️⃣ Cleaning up...")

        try:
            response = await client.post(
                f"{worker_manager_url}/workers/stop/{worker_id}"
            )
            if response.status_code == 200:
                print(f"✅ Worker {worker_id} stopped")
            else:
                print(f"⚠️  Failed to stop worker: {response.text}")
        except Exception as e:
            print(f"⚠️  Error stopping worker: {e}")

        print(
            "\n🎉 All tests passed! Local dynamic worker system is working correctly."
        )
        return True


async def main():
    """Main test function."""
    print("🚀 Local Dynamic Worker Test Suite")
    print("=" * 50)

    # Wait a moment for services to be ready
    print("⏳ Waiting for services to be ready...")
    await asyncio.sleep(3)

    success = await test_local_dynamic_workers()

    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
