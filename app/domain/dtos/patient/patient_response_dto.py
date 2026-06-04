"""DTO for returning patient information in API responses."""

from datetime import date
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class PatientResponseDTO(BaseModel):
    """Output DTO for patient-related API responses."""

    model_config = ConfigDict(from_attributes=True, json_schema_extra={
        "example": {
            "id": 1,
            "name": "João da Silva",
            "date_of_birth": "1985-03-22",
            "gender": "male",
            "contact_phone": "+55 11 99999-0000",
            "organization_name": "Hospital São Lucas",
        }
    })

    id: int = Field(..., description="Database primary key")
    name: str = Field(..., description="Full name of the patient")
    date_of_birth: date = Field(..., description="Date of birth")
    gender: str = Field(..., description="Patient gender")
    contact_phone: Optional[str] = Field(default=None, description="Contact phone number")
    organization_name: Optional[str] = Field(
        default=None, description="Hospital or clinic name"
    )
