"""Healthcare AI — FastAPI application entry point.

Run locally:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Environment variables are loaded from a ``.env`` file via pydantic-settings.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware import AppExceptionMiddleware, RequestLoggingMiddleware
from app.api.v1 import v1_router
from app.core.config import settings
from app.ml.models import ModelRegistry, create_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared state (models pre-loaded at startup)
# ---------------------------------------------------------------------------

_model_registry = ModelRegistry()


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load ML models on startup so the first request is fast."""
    logger.info("Starting Healthcare AI — loading ML models…")

    # Pre-warm the default inference models
    for model_type in ("ultra_light", "diagnostic"):
        try:
            model = create_model(model_type)
            _model_registry.register_model(model_type, model)
            logger.info("Loaded model: %s", model_type)
        except Exception as exc:
            logger.warning("Could not pre-load model '%s': %s", model_type, exc)

    logger.info("Healthcare AI is ready.")
    yield

    # Shutdown — nothing to clean up for in-process models
    logger.info("Healthcare AI shutting down.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "AI-assisted medical diagnostic API. "
        "Provides symptom-based inference with confidence scoring, "
        "differential diagnoses, and treatment recommendations."
    ),
    debug=settings.debug,
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware (order matters — outermost middleware runs first)
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to trusted origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AppExceptionMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(v1_router)


# ---------------------------------------------------------------------------
# Root redirect / liveness
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return {"message": f"Welcome to {settings.app_name}", "docs": "/docs"}
