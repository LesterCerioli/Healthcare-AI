"""Diagnosis request DTO — input payload sent to the diagnostic endpoint."""

from typing import Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class DiagnosisRequestDTO(BaseModel):
    """Input DTO for a diagnostic inference request.

    All numeric values should be normalised to the [0, 1] range:
      * 0.0 = symptom/marker absent
      * 1.0 = symptom fully present / marker at maximum severity

    Attributes
    ----------
    symptoms:
        Required mapping of symptom name → severity score.
        Example: ``{"fever": 0.8, "cough": 0.6}``.
    lab_markers:
        Optional mapping of lab-test name → normalised result.
        Example: ``{"blood_glucose": 0.9, "hba1c": 0.75}``.
    patient_context:
        Optional mapping of contextual features → normalised values.
        Example: ``{"age_normalised": 0.55, "bmi_normalised": 0.6}``.
    model_type:
        Which ML model to use for inference.  Defaults to ``"ultra_light"``.
        Available values: ``"ultra_light"``, ``"diagnostic"``,
        ``"diabetes"``, ``"cardiovascular"``, ``"symptom_analysis"``.
    """

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "symptoms": {"fever": 0.8, "cough": 0.6, "fatigue": 0.7},
            "lab_markers": {"blood_glucose": 0.85},
            "patient_context": {"age_normalised": 0.5},
            "model_type": "ultra_light",
        }
    })

    symptoms: Dict[str, float] = Field(
        ...,
        description="Symptom name → severity score [0, 1]",
    )
    lab_markers: Optional[Dict[str, float]] = Field(
        default=None,
        description="Lab marker name → normalised result [0, 1]",
    )
    patient_context: Optional[Dict[str, float]] = Field(
        default=None,
        description="Contextual feature name → normalised value [0, 1]",
    )
    model_type: str = Field(
        default="ultra_light",
        description="Model type to use for inference",
    )
