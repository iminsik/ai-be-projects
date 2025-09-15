#!/usr/bin/env python3
"""
Test script to verify that TensorFlow and Sklearn workers can be started properly.
"""

import asyncio
import subprocess
import sys
import os
import time


async def test_framework_worker(worker_type: str):
    """Test starting a worker of a specific framework type."""
    print(f"\n🧪 Testing {worker_type} worker...")

    # Change to ai-worker directory
    ai_worker_dir = os.path.join(os.path.dirname(__file__), "ai-worker")

    # Test dependency installation
    print(f"  📦 Installing dependencies for {worker_type}...")
    extra_map = {
        "tensorflow": "tensorflow",
        "sklearn": "sklearn",
        "pytorch-2.1": "pytorch_2_1",
    }

    if worker_type in extra_map:
        install_cmd = ["uv", "sync", "--extra", extra_map[worker_type]]
        result = subprocess.run(
            install_cmd,
            cwd=ai_worker_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"  ❌ Failed to install dependencies: {result.stderr}")
            return False
        else:
            print(f"  ✅ Dependencies installed successfully")

    # Test worker startup
    print(f"  🚀 Testing worker startup...")
    env = os.environ.copy()
    env.update(
        {
            "MODEL_FRAMEWORK": "tensorflow"
            if worker_type == "tensorflow"
            else "sklearn"
            if worker_type == "sklearn"
            else "pytorch",
            "WORKER_TYPE": worker_type,
            "REDIS_HOST": "localhost",
            "REDIS_PORT": "6379",
            "REDIS_DB": "0",
        }
    )

    # Start worker process
    process = subprocess.Popen(
        ["uv", "run", "python", "run_worker.py"],
        cwd=ai_worker_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait a bit for startup
    await asyncio.sleep(3)

    # Check if process is still running
    if process.poll() is None:
        print(f"  ✅ {worker_type} worker started successfully (PID: {process.pid})")
        process.terminate()
        process.wait()
        return True
    else:
        stdout, stderr = process.communicate()
        print(f"  ❌ {worker_type} worker failed to start:")
        print(f"     stdout: {stdout}")
        print(f"     stderr: {stderr}")
        return False


async def main():
    """Test all framework workers."""
    print("🔧 Testing Framework Worker Fixes")
    print("=" * 50)

    # Test frameworks
    frameworks = ["tensorflow", "sklearn", "pytorch-2.1"]
    results = {}

    for framework in frameworks:
        results[framework] = await test_framework_worker(framework)

    print("\n📊 Test Results:")
    print("=" * 50)
    for framework, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {framework:15} {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All framework workers are working correctly!")
    else:
        print("\n⚠️  Some framework workers are still failing.")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
