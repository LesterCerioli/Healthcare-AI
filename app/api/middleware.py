"""Custom FastAPI middleware for request logging and error handling."""

import logging
import time
import uuid

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.exceptions import AppException

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log each request with timing and a unique correlation ID."""

    async def dispatch(self, request: Request, call_next) -> Response:
        correlation_id = str(uuid.uuid4())
        start = time.perf_counter()

        # Attach to request state so downstream code can read it
        request.state.correlation_id = correlation_id

        logger.info(
            "→ %s %s [%s]",
            request.method,
            request.url.path,
            correlation_id,
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                "✗ %s %s [%s] %.1fms — unhandled: %s",
                request.method,
                request.url.path,
                correlation_id,
                elapsed,
                exc,
            )
            raise

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "← %s %s [%s] %d %.1fms",
            request.method,
            request.url.path,
            correlation_id,
            response.status_code,
            elapsed,
        )
        response.headers["X-Correlation-ID"] = correlation_id
        return response


class AppExceptionMiddleware(BaseHTTPMiddleware):
    """Convert AppException subclasses into structured JSON error responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            return await call_next(request)
        except AppException as exc:
            logger.warning("AppException: %s (status=%d)", exc.message, exc.status_code)
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": exc.message, "status_code": exc.status_code},
            )
