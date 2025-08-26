import os
from typing import Optional


class Config:
    """Configuration class for the AI worker service."""

    # Redis configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")

    # Model storage
    MODEL_STORAGE_PATH: str = os.getenv("MODEL_STORAGE_PATH", "./models")

    # Worker configuration
    WORKER_ID: str = os.getenv("WORKER_ID", f"worker-{os.getpid()}")
    MAX_CONCURRENT_JOBS: int = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
    JOB_TIMEOUT: int = int(os.getenv("JOB_TIMEOUT", "3600"))  # 1 hour

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # GPU configuration
    USE_GPU: bool = os.getenv("USE_GPU", "false").lower() == "true"
    GPU_DEVICE: str = os.getenv("GPU_DEVICE", "0")

    @classmethod
    def get_redis_url(cls) -> str:
        """Get Redis connection URL."""
        if cls.REDIS_PASSWORD:
            return f"redis://:{cls.REDIS_PASSWORD}@{cls.REDIS_HOST}:{cls.REDIS_PORT}/{cls.REDIS_DB}"
        return f"redis://{cls.REDIS_HOST}:{cls.REDIS_PORT}/{cls.REDIS_DB}"

    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production environment."""
        return cls.ENVIRONMENT.lower() == "production"

    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development environment."""
        return cls.ENVIRONMENT.lower() == "development"
