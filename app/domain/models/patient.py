"""Patient domain entity."""

from datetime import date
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class Patient(BaseModel):
    """Core patient entity used across the domain layer."""

    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = Field(default=None, description="Database primary key")
    name: str = Field(..., min_length=1, max_length=255, description="Full name of the patient")
    date_of_birth: date = Field(..., description="Date of birth (YYYY-MM-DD)")
    gender: str = Field(..., pattern="^(male|female|other)$", description="Patient gender")
    contact_phone: Optional[str] = Field(
        default=None,
        max_length=20,
        description="Contact phone number",
    )
    organization_name: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Hospital or clinic the patient belongs to",
    )
