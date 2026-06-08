"""Diagnostic service — orchestrates ML inference and persistence.

The service depends only on the IDiagnosisRepository contract and the ML
model registry, so it is completely decoupled from infrastructure details.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from app.domain.contracts.i_diagnosis_repository import IDiagnosisRepository
from app.domain.dtos.diagnosis.diagnosis_request_dto import DiagnosisRequestDTO
from app.domain.dtos.diagnosis.diagnosis_response_dto import DiagnosisResponseDTO
from app.domain.models.diagnosis import Diagnosis
from app.ml.models import ModelRegistry, create_model

logger = logging.getLogger(__name__)

# Supported model types — any value accepted by app.ml.models.create_model()
_VALID_MODEL_TYPES = {"ultra_light", "diagnostic", "diabetes", "cardiovascular", "symptom_analysis"}


class DiagnosticService:
    """Runs ML inference and optionally persists the result.

    Parameters
    ----------
    repository:
        An IDiagnosisRepository implementation injected at construction time.
        The service never imports a concrete repository class directly.
    model_registry:
        Optional pre-populated ModelRegistry.  When None a new registry is
        created.  Pass a shared registry from the application lifespan to
        avoid reloading models per request.
    persist_results:
        When True (default) the diagnosis is written to the database after
        successful inference.  Set to False in test / ephemeral modes.
    """

    def __init__(
        self,
        repository: IDiagnosisRepository,
        model_registry: Optional[ModelRegistry] = None,
        persist_results: bool = True,
    ) -> None:
        self._repository = repository
        self._registry = model_registry or ModelRegistry()
        self._persist = persist_results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def diagnose(
        self,
        dto: DiagnosisRequestDTO,
        patient_id: Optional[int] = None,
    ) -> DiagnosisResponseDTO:
        """Run inference for the given symptom payload and return a DTO.

        Parameters
        ----------
        dto:
            The validated request DTO from the API layer.
        patient_id:
            If supplied and ``persist_results`` is True the diagnosis entity
            is stored in the database with this patient FK.

        Returns
        -------
        DiagnosisResponseDTO
            Populated from the ML model output.  On inference failure the
            ``success`` flag is False and ``error`` contains a description.
        """
        model_type = dto.model_type if dto.model_type in _VALID_MODEL_TYPES else "ultra_light"

        # Resolve model — use registry cache or create on demand
        model = self._registry.get_model(model_type)
        if model is None:
            try:
                model = create_model(model_type)
                self._registry.register_model(model_type, model)
            except Exception as exc:
                logger.error("Failed to instantiate model '%s': %s", model_type, exc)
                return DiagnosisResponseDTO(
                    success=False,
                    error=f"Model '{model_type}' could not be loaded: {exc}",
                    model_type=model_type,
                )

        # Build the raw input dict the models expect
        input_data: dict = {"symptoms": dto.symptoms}
        if dto.lab_markers:
            input_data["lab_markers"] = dto.lab_markers
        if dto.patient_context:
            input_data["patient_context"] = dto.patient_context

        # Run inference
        try:
            raw_result: dict = model.predict(input_data)
        except Exception as exc:
            logger.error("Inference failed for model '%s': %s", model_type, exc)
            return DiagnosisResponseDTO(
                success=False,
                error=f"Inference error: {exc}",
                model_type=model_type,
            )

        if not raw_result.get("success", True):
            return DiagnosisResponseDTO(
                success=False,
                error=raw_result.get("error", "Inference returned no result"),
                model_type=model_type,
            )

        # Normalise output — both MLP and LLM models use slightly different keys
        primary = (
            raw_result.get("primary_diagnosis")
            or raw_result.get("diagnosis")
            or "Unknown"
        )
        urgency = raw_result.get("urgency_level", "Low")
        differential = raw_result.get("differential_diagnoses") or [
            {"condition": c.get("condition", c), "probability": c.get("probability", 0.0)}
            if isinstance(c, dict)
            else {"condition": str(c), "probability": 0.0}
            for c in raw_result.get("possible_conditions", [])
        ]
        examinations = raw_result.get("recommended_examinations") or []
        treatments = (
            raw_result.get("treatment_suggestions")
            or raw_result.get("recommendations")
            or []
        )
        disclaimer = raw_result.get(
            "clinical_disclaimer",
            (
                "These results are clinical decision support only. "
                "The attending physician is responsible for the final diagnosis and treatment plan."
            ),
        )

        response = DiagnosisResponseDTO(
            success=True,
            primary_diagnosis=primary,
            confidence=raw_result.get("confidence"),
            urgency_level=urgency,
            differential_diagnoses=differential,
            recommended_examinations=examinations,
            treatment_suggestions=treatments,
            clinical_disclaimer=disclaimer,
            model_type=model_type,
        )

        # Persist if a patient_id was provided and persistence is enabled
        if self._persist and patient_id is not None:
            await self._persist_diagnosis(response, patient_id)

        return response

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _persist_diagnosis(
        self, response: DiagnosisResponseDTO, patient_id: int
    ) -> None:
        """Write the diagnosis entity to the repository."""
        try:
            entity = Diagnosis(
                patient_id=patient_id,
                primary_diagnosis=response.primary_diagnosis or "Unknown",
                confidence=response.confidence or 0.0,
                urgency_level=response.urgency_level or "Low",
                differential_diagnoses=response.differential_diagnoses,
                recommended_examinations=response.recommended_examinations,
                treatment_suggestions=response.treatment_suggestions,
                created_at=datetime.now(timezone.utc),
            )
            await self._repository.create(entity)
        except Exception as exc:
            # Persistence failure must not break the API response
            logger.error("Failed to persist diagnosis for patient %d: %s", patient_id, exc)
