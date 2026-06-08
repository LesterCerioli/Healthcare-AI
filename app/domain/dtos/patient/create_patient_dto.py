"""DTO for creating a new patient record."""

from datetime import date
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class CreatePatientDTO(BaseModel):
    """Input DTO for the patient creation endpoint."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "name": "João da Silva",
            "date_of_birth": "1985-03-22",
            "gender": "male",
            "contact_phone": "+55 11 99999-0000",
            "organization_name": "Hospital São Lucas",
        }
    })

    name: str = Field(..., min_length=1, max_length=255, description="Full name of the patient")
    date_of_birth: date = Field(..., description="Date of birth (YYYY-MM-DD)")
    gender: str = Field(
        ...,
        pattern="^(male|female|other)$",
        description="Patient gender: male, female, or other",
    )
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
