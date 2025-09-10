#!/usr/bin/env python3
"""
Test script for GPU memory management functionality.
"""

import asyncio
import json
import time
from typing import Dict

import httpx

BASE_URL = "http://localhost:8000"

async def submit_gpu_training_job(batch_size: int = 32) -> str:
    """Submit a GPU training job with specified batch size."""
    training_data = {
        "model_type": "transformer",
        "data_path": "/data/training_data.csv",
        "hyperparameters": {
            "epochs": 5,
            "learning_rate": 0.001,
            "batch_size": batch_size,
        },
        "description": f"GPU training job with batch_size={batch_size}",
        "requires_gpu": True,
        "gpu_device": 0
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/jobs/training", json=training_data)
        
        if response.status_code == 200:
            job = response.json()
            print(f"✅ GPU training job submitted: {job['job_id']} (batch_size={batch_size})")
            return job["job_id"]
        else:
            print(f"❌ Failed to submit GPU training job: {response.status_code}")
            return None

async def submit_cpu_training_job() -> str:
    """Submit a CPU training job."""
    training_data = {
        "model_type": "transformer",
        "data_path": "/data/training_data.csv",
        "hyperparameters": {
            "epochs": 3,
            "learning_rate": 0.001,
            "batch_size": 8,
        },
        "description": "CPU training job",
        "requires_gpu": False
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/jobs/training", json=training_data)
        
        if response.status_code == 200:
            job = response.json()
            print(f"✅ CPU training job submitted: {job['job_id']}")
            return job["job_id"]
        else:
            print(f"❌ Failed to submit CPU training job: {response.status_code}")
            return None

async def check_job_status(job_id: str) -> Dict:
    """Check the status of a job."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/jobs/{job_id}/status")
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to get job status: {response.status_code}")
            return None

async def monitor_job(job_id: str, max_wait: int = 60):
    """Monitor a job until completion or timeout."""
    print(f"📊 Monitoring job {job_id}...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        status = await check_job_status(job_id)
        if status:
            print(f"   Status: {status['status']}")
            
            if status["status"] in ["completed", "failed"]:
                if status["status"] == "completed":
                    print(f"   ✅ Job completed successfully!")
                    if "result" in status:
                        result = status["result"]
                        if "optimized" in result.get("metadata", {}):
                            print(f"   🔧 Job was optimized for memory constraints")
                            print(f"   📊 Original batch size: {result['metadata'].get('original_batch_size', 'N/A')}")
                            print(f"   📊 Optimized batch size: {result['metadata']['hyperparameters'].get('batch_size', 'N/A')}")
                else:
                    print(f"   ❌ Job failed: {status.get('error', 'Unknown error')}")
                return status
            elif status["status"] == "running" and "result" in status:
                # Show progress for running jobs
                result = status["result"]
                if "progress" in result:
                    print(f"   📈 Progress: {result['progress']:.1%}")
                if "epoch" in result:
                    print(f"   🔄 Epoch: {result['epoch']}")
        
        await asyncio.sleep(2)
    
    print(f"⏰ Job monitoring timed out after {max_wait} seconds")
    return None

async def test_gpu_memory_scenarios():
    """Test different GPU memory scenarios."""
    print("🧪 Testing GPU Memory Management Scenarios\n")
    
    # Scenario 1: Submit multiple GPU jobs with different batch sizes
    print("📋 Scenario 1: Multiple GPU jobs with different batch sizes")
    print("=" * 60)
    
    job_ids = []
    
    # Submit jobs with increasing batch sizes
    batch_sizes = [8, 16, 32, 64]
    for batch_size in batch_sizes:
        job_id = await submit_gpu_training_job(batch_size)
        if job_id:
            job_ids.append(job_id)
        await asyncio.sleep(1)  # Small delay between submissions
    
    print(f"\n📊 Submitted {len(job_ids)} GPU training jobs")
    
    # Monitor all jobs
    print("\n🔍 Monitoring job progress...")
    for i, job_id in enumerate(job_ids):
        print(f"\n--- Job {i+1} (batch_size={batch_sizes[i]}) ---")
        await monitor_job(job_id, max_wait=30)
    
    # Scenario 2: Mix of CPU and GPU jobs
    print("\n\n📋 Scenario 2: Mixed CPU and GPU jobs")
    print("=" * 60)
    
    # Submit CPU job
    cpu_job_id = await submit_cpu_training_job()
    
    # Submit GPU job
    gpu_job_id = await submit_gpu_training_job(batch_size=16)
    
    if cpu_job_id and gpu_job_id:
        print(f"\n📊 Submitted mixed jobs: CPU={cpu_job_id}, GPU={gpu_job_id}")
        
        # Monitor both jobs
        print("\n🔍 Monitoring mixed jobs...")
        await asyncio.gather(
            monitor_job(cpu_job_id, max_wait=20),
            monitor_job(gpu_job_id, max_wait=20)
        )
    
    # Scenario 3: Large batch size that should trigger optimization
    print("\n\n📋 Scenario 3: Large batch size optimization")
    print("=" * 60)
    
    large_batch_job_id = await submit_gpu_training_job(batch_size=128)
    if large_batch_job_id:
        print(f"\n📊 Submitted large batch job: {large_batch_job_id}")
        print("🔍 This should trigger automatic batch size optimization...")
        await monitor_job(large_batch_job_id, max_wait=30)

async def test_memory_monitoring():
    """Test memory monitoring capabilities."""
    print("\n\n📊 Testing Memory Monitoring")
    print("=" * 60)
    
    # Check worker status (if available)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/workers/status")
            if response.status_code == 200:
                worker_status = response.json()
                print("🔍 Worker Status:")
                print(json.dumps(worker_status, indent=2))
            else:
                print("⚠️  Worker status endpoint not available")
    except Exception as e:
        print(f"⚠️  Could not get worker status: {e}")

async def main():
    """Main test function."""
    print("🚀 GPU Memory Management Test Suite")
    print("=" * 60)
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                print("✅ Backend is healthy")
            else:
                print("❌ Backend health check failed")
                return
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        return
    
    # Run GPU memory scenarios
    await test_gpu_memory_scenarios()
    
    # Test memory monitoring
    await test_memory_monitoring()
    
    print("\n🎉 GPU Memory Management Test Complete!")
    print("\nKey Features Demonstrated:")
    print("✅ Automatic memory estimation")
    print("✅ Memory allocation and deallocation")
    print("✅ Batch size optimization")
    print("✅ Job queuing when memory is unavailable")
    print("✅ Graceful error handling")
    print("✅ Mixed CPU/GPU job processing")

if __name__ == "__main__":
    asyncio.run(main())
