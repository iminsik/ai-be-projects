#!/usr/bin/env python3
"""
Test script for the frontend application.
This script tests the frontend by making HTTP requests to verify it's working.
"""

import requests
import time
import sys
from urllib.parse import urljoin


def test_frontend():
    """Test the frontend application."""
    frontend_url = "http://localhost:3000"
    backend_url = "http://localhost:8000"

    print("🧪 Testing Frontend Application...")
    print(f"Frontend URL: {frontend_url}")
    print(f"Backend URL: {backend_url}")
    print()

    # Test 1: Check if frontend is accessible
    print("1. Testing frontend accessibility...")
    try:
        response = requests.get(frontend_url, timeout=10)
        if response.status_code == 200:
            print("   ✅ Frontend is accessible")
        else:
            print(f"   ❌ Frontend returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Frontend is not accessible: {e}")
        return False

    # Test 2: Check if backend is accessible
    print("2. Testing backend accessibility...")
    try:
        response = requests.get(f"{backend_url}/health", timeout=10)
        if response.status_code == 200:
            print("   ✅ Backend is accessible")
            health_data = response.json()
            print(f"   📊 Backend status: {health_data.get('status', 'unknown')}")
        else:
            print(f"   ❌ Backend returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Backend is not accessible: {e}")
        return False

    # Test 3: Test API proxy through frontend
    print("3. Testing API proxy through frontend...")
    try:
        # Test health endpoint through frontend proxy
        response = requests.get(f"{frontend_url}/api/health", timeout=10)
        if response.status_code == 200:
            print("   ✅ API proxy is working")
            health_data = response.json()
            print(f"   📊 API status: {health_data.get('status', 'unknown')}")
        else:
            print(f"   ❌ API proxy returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ API proxy failed: {e}")
        return False

    # Test 4: Test frameworks endpoint
    print("4. Testing frameworks endpoint...")
    try:
        response = requests.get(f"{frontend_url}/api/frameworks", timeout=10)
        if response.status_code == 200:
            print("   ✅ Frameworks endpoint is working")
            frameworks_data = response.json()
            frameworks = frameworks_data.get("frameworks", [])
            print(f"   📊 Available frameworks: {len(frameworks)}")
            for framework in frameworks[:3]:  # Show first 3
                print(f"      - {framework}")
            if len(frameworks) > 3:
                print(f"      ... and {len(frameworks) - 3} more")
        else:
            print(f"   ❌ Frameworks endpoint returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Frameworks endpoint failed: {e}")
        return False

    # Test 5: Test workers status endpoint
    print("5. Testing workers status endpoint...")
    try:
        response = requests.get(f"{frontend_url}/api/workers/status", timeout=10)
        if response.status_code == 200:
            print("   ✅ Workers status endpoint is working")
            workers_data = response.json()
            total_workers = workers_data.get("total_workers", 0)
            available_workers = workers_data.get("available_workers", 0)
            print(f"   📊 Total workers: {total_workers}")
            print(f"   📊 Available workers: {available_workers}")
        else:
            print(
                f"   ❌ Workers status endpoint returned status {response.status_code}"
            )
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Workers status endpoint failed: {e}")
        return False

    print()
    print("🎉 All frontend tests passed!")
    print()
    print("🌐 Frontend is ready at: http://localhost:3000")
    print("🔧 Backend API is ready at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print()
    print("💡 You can now:")
    print("   - Open http://localhost:3000 in your browser")
    print("   - Submit training and inference jobs through the web interface")
    print("   - Monitor job progress in real-time")
    print("   - Check system status and worker availability")

    return True


if __name__ == "__main__":
    print("AI Job Queue System - Frontend Test")
    print("=" * 50)
    print()

    # Wait a moment for services to start
    print("⏳ Waiting for services to start...")
    time.sleep(3)

    success = test_frontend()

    if not success:
        print()
        print("❌ Frontend tests failed!")
        print("💡 Make sure both frontend and backend are running:")
        print("   - Frontend: npm run dev (in frontend/ directory)")
        print("   - Backend: ./scripts/run_local_dynamic.sh")
        sys.exit(1)
    else:
        print()
        print("✅ Frontend is working correctly!")
        sys.exit(0)
