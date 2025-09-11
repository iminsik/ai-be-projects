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

    # Test framework-specific dependencies
    print("\nTesting framework-specific dependencies...")
    frameworks = [
        ("pytorch_2_0", "PyTorch 2.0"),
        ("pytorch_2_1", "PyTorch 2.1"),
        ("tensorflow", "TensorFlow"),
        ("sklearn", "Scikit-learn"),
        ("pytorch_2_0_gpu", "PyTorch 2.0 GPU"),
        ("pytorch_2_1_gpu", "PyTorch 2.1 GPU"),
    ]

    for extra_name, display_name in frameworks:
        # Check if the extra is defined in pyproject.toml
        pyproject_path = os.path.join(worker_dir, "pyproject.toml")
        if os.path.exists(pyproject_path):
            with open(pyproject_path, "r") as f:
                content = f.read()
                if extra_name in content:
                    print(f"✅ {display_name} dependencies configured")
                else:
                    print(f"⚠️  {display_name} dependencies not found in pyproject.toml")


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
        print("2. Install AI worker dependencies:")
        print("   cd ai-worker")
        print("   uv sync --extra pytorch_2_1  # or your preferred framework")
        print("3. Run the system: ./scripts/run_local.sh")
        print("4. Test the API: python scripts/test_api.py")
        print("\nAvailable frameworks:")
        print("   - PyTorch 2.0: uv sync --extra pytorch_2_0")
        print("   - PyTorch 2.1: uv sync --extra pytorch_2_1")
        print("   - TensorFlow: uv sync --extra tensorflow")
        print("   - Scikit-learn: uv sync --extra sklearn")
        print("   - GPU PyTorch 2.0: uv sync --extra pytorch_2_0_gpu")
        print("   - GPU PyTorch 2.1: uv sync --extra pytorch_2_1_gpu")
    else:
        print("❌ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
