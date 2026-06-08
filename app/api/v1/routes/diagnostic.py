"""Diagnostic inference endpoint.

POST /api/v1/diagnostic — accepts a DiagnosisRequestDTO and returns a
DiagnosisResponseDTO produced by the DiagnosticService.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.dtos.diagnosis.diagnosis_request_dto import DiagnosisRequestDTO
from app.domain.dtos.diagnosis.diagnosis_response_dto import DiagnosisResponseDTO
from app.infrastructure.database.connection import get_db_session
from app.infrastructure.repositories.diagnosis_repository import DiagnosisRepository
from app.services.diagnostic_service import DiagnosticService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["diagnostic"])


# ---------------------------------------------------------------------------
# Dependency: build the service with a request-scoped DB session
# ---------------------------------------------------------------------------

def get_diagnostic_service(
    session: AsyncSession = Depends(get_db_session),
) -> DiagnosticService:
    """FastAPI dependency that wires the service to a fresh DB session."""
    repository = DiagnosisRepository(session)
    return DiagnosticService(repository=repository, persist_results=True)


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post(
    "/diagnostic",
    response_model=DiagnosisResponseDTO,
    summary="Run AI diagnostic inference",
    description=(
        "Accepts a set of symptoms (and optional lab markers / patient context) "
        "and returns an AI-assisted diagnosis with confidence scores, differential "
        "diagnoses, recommended examinations, and treatment suggestions."
    ),
)
async def run_diagnostic(
    request: DiagnosisRequestDTO,
    patient_id: Optional[int] = Query(
        default=None,
        description="Patient primary key. When provided the diagnosis is persisted.",
    ),
    service: DiagnosticService = Depends(get_diagnostic_service),
) -> DiagnosisResponseDTO:
    """Run diagnostic inference for the supplied symptom payload."""
    logger.info(
        "Diagnostic request — model_type=%s symptoms=%s",
        request.model_type,
        list(request.symptoms.keys()),
    )
    return await service.diagnose(dto=request, patient_id=patient_id)
