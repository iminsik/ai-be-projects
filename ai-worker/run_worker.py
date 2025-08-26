#!/usr/bin/env python3
"""
Simple runner script for the AI worker.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from worker import main

if __name__ == "__main__":
    asyncio.run(main())
