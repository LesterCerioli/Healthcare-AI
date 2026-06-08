"""Route modules for API v1."""

from .health import router as health_router
from .diagnostic import router as diagnostic_router

__all__ = ["health_router", "diagnostic_router"]
