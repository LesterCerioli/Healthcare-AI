"""Diagnosis domain entity."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class Diagnosis(BaseModel):
    """Represents a completed diagnostic assessment for a patient."""

    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = Field(default=None, description="Database primary key")
    patient_id: int = Field(..., description="FK to the patient this diagnosis belongs to")
    primary_diagnosis: str = Field(..., description="Most likely diagnostic condition")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score [0, 1]")
    urgency_level: str = Field(
        ...,
        description="Clinical urgency: 'Low', 'Medium', or 'High'",
    )
    differential_diagnoses: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ranked list of alternative diagnoses with probabilities",
    )
    recommended_examinations: List[str] = Field(
        default_factory=list,
        description="Suggested clinical examinations or lab tests",
    )
    treatment_suggestions: List[str] = Field(
        default_factory=list,
        description="Initial treatment or management recommendations",
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the diagnosis was created",
    )
