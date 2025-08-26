#!/usr/bin/env python3
"""
Main entry point for the AI worker module.
"""

import asyncio
from .worker import main

if __name__ == "__main__":
    asyncio.run(main())
