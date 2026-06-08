"""Data Transfer Objects for API input/output."""

from .diagnosis.diagnosis_request_dto import DiagnosisRequestDTO
from .diagnosis.diagnosis_response_dto import DiagnosisResponseDTO
from .patient.create_patient_dto import CreatePatientDTO
from .patient.patient_response_dto import PatientResponseDTO

__all__ = [
    "DiagnosisRequestDTO",
    "DiagnosisResponseDTO",
    "CreatePatientDTO",
    "PatientResponseDTO",
]
