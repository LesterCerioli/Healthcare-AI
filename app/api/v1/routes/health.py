"""Health-check endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service status and API version.",
)
async def health_check() -> HealthResponse:
    """Lightweight liveness probe used by load balancers and orchestrators."""
    return HealthResponse(status="ok", version="1.0.0")
