"""API v1 package — aggregates all route modules."""

from fastapi import APIRouter

from .routes.health import router as health_router
from .routes.diagnostic import router as diagnostic_router

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(health_router)
v1_router.include_router(diagnostic_router)

__all__ = ["v1_router"]
