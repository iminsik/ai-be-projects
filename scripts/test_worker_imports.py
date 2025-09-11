#!/usr/bin/env python3
"""
Test script to verify worker imports work correctly.
"""

import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ai-worker", "src"))


def test_imports():
    """Test that all imports work correctly."""
    print("🧪 Testing worker imports...")

    try:
        # Test basic imports
        print("  ✓ Testing basic imports...")
        from config import Config

        print("  ✓ Config imported successfully")

        from gpu_manager import GPUMemoryManager, JobQueueManager

        print("  ✓ GPU manager imported successfully")

        from worker import import_framework_libraries

        print("  ✓ Worker imported successfully")

        # Test framework imports
        print("  ✓ Testing framework imports...")

        # Test PyTorch import
        os.environ["MODEL_FRAMEWORK"] = "pytorch"
        os.environ["WORKER_TYPE"] = "pytorch-2.1"

        framework, lib = import_framework_libraries()
        if framework:
            print(f"  ✓ PyTorch framework loaded: {framework}")
        else:
            print("  ⚠ PyTorch not available (expected if not installed)")

        # Test TensorFlow import
        os.environ["MODEL_FRAMEWORK"] = "tensorflow"
        os.environ["WORKER_TYPE"] = "tensorflow"

        framework, lib = import_framework_libraries()
        if framework:
            print(f"  ✓ TensorFlow framework loaded: {framework}")
        else:
            print("  ⚠ TensorFlow not available (expected if not installed)")

        # Test Scikit-learn import
        os.environ["MODEL_FRAMEWORK"] = "sklearn"
        os.environ["WORKER_TYPE"] = "sklearn"

        framework, lib = import_framework_libraries()
        if framework:
            print(f"  ✓ Scikit-learn framework loaded: {framework}")
        else:
            print("  ⚠ Scikit-learn not available (expected if not installed)")

        print("✅ All imports tested successfully!")
        return True

    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
