"""Symptom domain entity."""

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class Symptom(BaseModel):
    """Represents a single clinical symptom with an optional severity score."""

    model_config = ConfigDict(from_attributes=True)

    name: str = Field(..., min_length=1, max_length=100, description="Symptom identifier key")
    severity: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Symptom severity normalised to [0, 1]; 1.0 = fully present",
    )
    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Free-text clinical description of the symptom",
    )
