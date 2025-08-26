#!/usr/bin/env python3
"""
Basic test script to verify the system setup without Redis.
"""

import sys
import os


def test_backend_imports():
    """Test backend imports."""
    print("Testing backend imports...")
    try:
        # Add backend src to path
        backend_src = os.path.join(os.path.dirname(__file__), "..", "backend", "src")
        sys.path.insert(0, backend_src)

        from config import Config

        print(f"✅ Backend config loaded: REDIS_HOST={Config.REDIS_HOST}")

        from main import app

        print("✅ Backend FastAPI app imported successfully")

        return True
    except Exception as e:
        print(f"❌ Backend import failed: {e}")
        return False


def test_ai_worker_imports():
    """Test AI worker imports."""
    print("Testing AI worker imports...")
    try:
        # Add ai-worker src to path
        worker_src = os.path.join(os.path.dirname(__file__), "..", "ai-worker", "src")
        sys.path.insert(0, worker_src)

        from config import Config

        print(
            f"✅ AI worker config loaded: MODEL_STORAGE_PATH={Config.MODEL_STORAGE_PATH}"
        )

        from worker import AIWorker

        print("✅ AI worker class imported successfully")

        return True
    except Exception as e:
        print(f"❌ AI worker import failed: {e}")
        return False


def test_uv_dependencies():
    """Test uv dependency installation."""
    print("Testing uv dependencies...")

    # Test backend dependencies
    backend_dir = os.path.join(os.path.dirname(__file__), "..", "backend")
    if os.path.exists(os.path.join(backend_dir, ".venv")):
        print("✅ Backend virtual environment exists")
    else:
        print("⚠️  Backend virtual environment not found")

    # Test ai-worker dependencies
    worker_dir = os.path.join(os.path.dirname(__file__), "..", "ai-worker")
    if os.path.exists(os.path.join(worker_dir, ".venv")):
        print("✅ AI worker virtual environment exists")
    else:
        print("⚠️  AI worker virtual environment not found")


def main():
    """Run all tests."""
    print("🧪 Basic System Test\n")

    backend_ok = test_backend_imports()
    print()

    worker_ok = test_ai_worker_imports()
    print()

    test_uv_dependencies()
    print()

    if backend_ok and worker_ok:
        print("🎉 All basic tests passed!")
        print("\nNext steps:")
        print("1. Install Redis: brew install redis && brew services start redis")
        print("2. Run the system: ./scripts/run_local.sh")
        print("3. Test the API: python scripts/test_api.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
