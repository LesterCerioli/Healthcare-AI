"""Diagnosis response DTO — output returned by the diagnostic endpoint."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class DiagnosisResponseDTO(BaseModel):
    """Output DTO for a completed diagnostic inference.

    Field names mirror the output dict produced by BaseMedicalLLM.postprocess()
    and by UltraLightMedicalModel.postprocess() so that the service layer can
    map model output directly without manual field translation.
    """

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": True,
            "primary_diagnosis": "Influenza",
            "confidence": 0.87,
            "urgency_level": "Medium",
            "differential_diagnoses": [
                {"condition": "Influenza", "probability": 0.87},
                {"condition": "Common Cold", "probability": 0.09},
                {"condition": "COVID-19", "probability": 0.04},
            ],
            "recommended_examinations": ["Complete blood count", "Rapid influenza test"],
            "treatment_suggestions": ["Antiviral medication if early", "Rest and fluids"],
            "clinical_disclaimer": (
                "These results are clinical decision support only. "
                "The attending physician is responsible for the final diagnosis."
            ),
            "model_type": "ultra_light",
        }
    })

    success: bool = Field(default=True, description="Whether inference completed successfully")
    primary_diagnosis: Optional[str] = Field(
        default=None,
        description="Most likely diagnostic condition",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Model confidence for the primary diagnosis [0, 1]",
    )
    urgency_level: Optional[str] = Field(
        default=None,
        description="Clinical urgency: Low / Medium / High",
    )
    differential_diagnoses: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top-k alternative diagnoses with probabilities",
    )
    recommended_examinations: List[str] = Field(
        default_factory=list,
        description="Suggested examinations or lab tests",
    )
    treatment_suggestions: List[str] = Field(
        default_factory=list,
        description="Initial treatment or management recommendations",
    )
    clinical_disclaimer: Optional[str] = Field(
        default=(
            "These results are clinical decision support only. "
            "The attending physician is responsible for the final diagnosis and treatment plan."
        ),
        description="Mandatory disclaimer for clinical AI output",
    )
    model_type: Optional[str] = Field(
        default=None,
        description="The model type that produced this response",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message when success is False",
    )
