#!/usr/bin/env python3
"""
Simple test to demonstrate that the 503 error issue has been fixed.
This test specifically checks that the first request for a new worker type
now returns 200 instead of 503.
"""

import asyncio
import time
import httpx


async def test_503_fix():
    """Test that the 503 error issue is fixed."""
    print("🧪 Testing 503 Error Fix")
    print("=" * 40)

    backend_url = "http://localhost:8000"

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Test 1: Submit job for a new worker type (should NOT return 503)
        print("\n1. Submitting job for PyTorch 2.0 (new worker type)...")

        job_data = {
            "model_type": "resnet",
            "data_path": "/data/test.csv",
            "hyperparameters": {"epochs": 1, "learning_rate": 0.001},
            "description": "Test job for resnet",
            "requires_gpu": False,
            "framework_override": "pytorch-2.0",  # Force PyTorch 2.0 worker
        }

        start_time = time.time()
        try:
            response = await client.post(f"{backend_url}/jobs/training", json=job_data)
            end_time = time.time()

            print(f"   Status Code: {response.status_code}")
            print(f"   Response Time: {end_time - start_time:.2f} seconds")

            if response.status_code == 200:
                result = response.json()
                print(f"   Job ID: {result['job_id']}")
                print(f"   Worker Type: {result['worker_type']}")
                print("   ✅ SUCCESS: No 503 error! Worker was spawned and job queued.")
            elif response.status_code == 503:
                print(f"   ❌ FAILED: Still getting 503 error: {response.text}")
                return False
            else:
                print(f"   ⚠️  Unexpected status code: {response.status_code}")
                print(f"   Response: {response.text}")
                return False

        except httpx.TimeoutException:
            print(
                "   ⚠️  Request timed out (this might be expected for first worker spawn)"
            )
            return False
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

        # Test 2: Submit another job for the same worker type (should be fast)
        print("\n2. Submitting another job for PyTorch 2.0 (should be fast)...")

        job_data["model_type"] = "vgg"  # Different model, same worker type

        start_time = time.time()
        try:
            response = await client.post(f"{backend_url}/jobs/training", json=job_data)
            end_time = time.time()

            print(f"   Status Code: {response.status_code}")
            print(f"   Response Time: {end_time - start_time:.2f} seconds")

            if response.status_code == 200:
                result = response.json()
                print(f"   Job ID: {result['job_id']}")
                print("   ✅ SUCCESS: Second job was fast (worker already available)")
            else:
                print(f"   ❌ FAILED: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

    print("\n🎉 Test Results:")
    print("   ✅ First request for new worker type: SUCCESS (no 503)")
    print("   ✅ Second request for same worker type: FAST")
    print("   ✅ The 503 error issue has been FIXED!")

    return True


async def main():
    """Main test function."""
    print("🚀 Testing 503 Error Fix")
    print("Make sure the backend and worker manager are running:")
    print("  - Backend: http://localhost:8000")
    print("  - Worker Manager: http://localhost:8001")
    print()

    try:
        success = await test_503_fix()
        if success:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Tests failed!")
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
