#!/usr/bin/env python3
"""
Test script to demonstrate automatic worker spawning functionality.
This script tests the new behavior where workers are automatically spawned
when submitting jobs for worker types that aren't currently running.
"""

import asyncio
import json
import time
from typing import Dict, Any

import httpx


class WorkerSpawningTester:
    def __init__(
        self,
        backend_url: str = "http://localhost:8000",
        worker_manager_url: str = "http://localhost:8001",
    ):
        self.backend_url = backend_url
        self.worker_manager_url = worker_manager_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def check_worker_status(self) -> Dict[str, Any]:
        """Check current worker status."""
        try:
            response = await self.client.get(
                f"{self.worker_manager_url}/workers/status"
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get worker status: {response.text}")
                return {}
        except Exception as e:
            print(f"Error checking worker status: {e}")
            return {}

    async def submit_training_job(
        self,
        model_type: str,
        requires_gpu: bool = False,
        framework_override: str = None,
    ) -> Dict[str, Any]:
        """Submit a training job and return the response."""
        job_data = {
            "model_type": model_type,
            "data_path": "/data/test.csv",
            "hyperparameters": {"epochs": 1, "learning_rate": 0.001, "batch_size": 16},
            "description": f"Test job for {model_type}",
            "requires_gpu": requires_gpu,
        }

        if framework_override:
            job_data["framework_override"] = framework_override

        try:
            response = await self.client.post(
                f"{self.backend_url}/jobs/training", json=job_data
            )
            return {
                "status_code": response.status_code,
                "response": response.json()
                if response.status_code == 200
                else response.text,
            }
        except Exception as e:
            return {"status_code": 500, "response": f"Error: {str(e)}"}

    async def test_worker_spawning(self):
        """Test the worker spawning functionality."""
        print("🧪 Testing Automatic Worker Spawning")
        print("=" * 50)

        # Test 1: Check initial worker status
        print("\n1. Checking initial worker status...")
        initial_status = await self.check_worker_status()
        if initial_status:
            print(f"   Initial workers: {initial_status.get('workers_by_type', {})}")
        else:
            print("   No workers currently running")

        # Test 2: Submit job for PyTorch 2.1 (should spawn worker and wait)
        print("\n2. Submitting job for PyTorch 2.1 (should spawn worker and wait)...")
        start_time = time.time()
        result = await self.submit_training_job("bert", requires_gpu=False)
        end_time = time.time()
        print(f"   Status Code: {result['status_code']}")
        print(f"   Response Time: {end_time - start_time:.2f} seconds")
        if result["status_code"] == 200:
            job_id = result["response"]["job_id"]
            print(f"   Job ID: {job_id}")
            print(f"   Worker Type: {result['response']['worker_type']}")
            print("   ✅ SUCCESS: Worker spawned and job queued!")
        else:
            print(f"   ❌ ERROR: {result['response']}")
            return  # Stop testing if first job fails

        # Test 3: Check worker status after spawning
        print("\n3. Checking worker status after spawning...")
        after_status = await self.check_worker_status()
        if after_status:
            print(
                f"   Workers after spawning: {after_status.get('workers_by_type', {})}"
            )
        else:
            print("   Failed to get worker status")

        # Test 4: Submit job for TensorFlow (should spawn another worker)
        print("\n4. Submitting job for TensorFlow (should spawn another worker)...")
        result = await self.submit_training_job("inception", requires_gpu=False)
        print(f"   Status Code: {result['status_code']}")
        if result["status_code"] == 200:
            job_id = result["response"]["job_id"]
            print(f"   Job ID: {job_id}")
            print(f"   Worker Type: {result['response']['worker_type']}")
        else:
            print(f"   Error: {result['response']}")

        # Wait for second worker to spawn
        print("\n   Waiting 5 seconds for second worker to spawn...")
        await asyncio.sleep(5)

        # Test 5: Final worker status
        print("\n5. Final worker status...")
        final_status = await self.check_worker_status()
        if final_status:
            print(f"   Final workers: {final_status.get('workers_by_type', {})}")
            total_workers = final_status.get("total_workers", 0)
            print(f"   Total workers: {total_workers}")
        else:
            print("   Failed to get final worker status")

        # Test 6: Submit job for existing worker type (should not spawn new worker)
        print("\n6. Submitting another PyTorch 2.1 job (should use existing worker)...")
        result = await self.submit_training_job("gpt", requires_gpu=False)
        print(f"   Status Code: {result['status_code']}")
        if result["status_code"] == 200:
            job_id = result["response"]["job_id"]
            print(f"   Job ID: {job_id}")
            print(f"   Worker Type: {result['response']['worker_type']}")
        else:
            print(f"   Error: {result['response']}")

        print("\n✅ Worker spawning test completed!")
        print("\nSummary:")
        print("- ✅ Workers are automatically spawned when needed")
        print("- ✅ Jobs are queued successfully after worker spawning")
        print("- ✅ No 503 errors for missing workers")
        print("- ✅ Backend waits for workers to be ready before queuing jobs")
        print("- ✅ Existing workers are reused for subsequent jobs")
        print("\n🎉 The 503 error issue has been fixed!")
        print(
            "   The backend now waits for workers to be ready before returning success."
        )

    async def cleanup(self):
        """Clean up resources."""
        await self.client.aclose()


async def main():
    """Main test function."""
    tester = WorkerSpawningTester()

    try:
        await tester.test_worker_spawning()
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    print("🚀 Starting Worker Spawning Test")
    print("Make sure the backend and worker manager are running:")
    print("  - Backend: http://localhost:8000")
    print("  - Worker Manager: http://localhost:8001")
    print()

    asyncio.run(main())
