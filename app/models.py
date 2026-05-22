from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from uuid import UUID

class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"

class Urgency(str, Enum):
    EMERGENCY = "emergency"
    URGENT = "urgent"
    PRIORITY = "priority"
    ROUTINE = "routine"
    MONITORING = "monitoring"

class TreatmentPhase(str, Enum):
    IMMEDIATE = "immediate"
    ACUTE = "acute"
    SUBACUTE = "subacute"
    MAINTENANCE = "maintenance"
    PREVENTION = "prevention"

class Language(str, Enum):
    ENGLISH = "en"
    PORTUGUESE = "pt"
    SPANISH = "es"

@dataclass
class BilingualText:
    en: str
    pt: str
    
    def get(self, lang: Language) -> str:
        if lang == Language.ENGLISH:
            return self.en
        elif lang == Language.PORTUGUESE:
            return self.pt
        return self.en  # Default to English
    
    def to_dict(self) -> Dict[str, str]:
        return {"en": self.en, "pt": self.pt}

@dataclass
class Symptom:
    """Clinical symptom - internal matching logic only"""
    id: str
    name: BilingualText
    synonyms: List[BilingualText]
    severity_weight: float
    typical_duration_days: Optional[float] = None
    red_flags: List[BilingualText] = field(default_factory=list)
    icd11_code: Optional[str] = None

@dataclass
class MedicalCondition:
    """Medical condition with ICD-10 as natural key"""
    id: str
    name: BilingualText
    icd10_code: str
    icd11_code: Optional[str] = None
    symptoms: List[str]
    symptom_weights: Dict[str, float]
    differentials: List[BilingualText]
    required_exams: List[BilingualText]
    severity: Severity
    urgency: Urgency
    risk_factors: List[BilingualText]
    complications: List[BilingualText]
    treatment_protocols: Dict[TreatmentPhase, List[BilingualText]]
    ai_confidence_threshold: float = 0.65

# API Request/Response Models
class DiagnosisRequest(BaseModel):
    text: str
    session_identifier: Optional[str] = None
    language: Optional[Language] = Language.ENGLISH
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    known_conditions: Optional[List[str]] = None
    organization_id: UUID  # Required for SaaS multi-tenancy

class DiagnosisResponse(BaseModel):
    session_id: UUID
    session_identifier: str
    diagnostic_result_id: UUID
    organization_id: UUID
    diagnosis: str
    confidence: float
    severity: str
    urgency: str
    recommendation: str
    treatment_priority: int
    processing_time_ms: float
    created_at: datetime

class OrganizationSettings(BaseModel):
    organization_id: UUID
    custom_treatments_enabled: bool = False
    preferred_language: Language = Language.ENGLISH
    ai_confidence_threshold: float = 0.65
    audit_enabled: bool = True

@dataclass
class DiagnosticResult:
    """Diagnostic result with organization isolation"""
    organization_id: UUID
    session_id: UUID
    condition_id: str
    condition_name: BilingualText
    confidence: float
    severity: Severity
    urgency: Urgency
    matched_symptoms: List[Dict]
    differential_diagnoses: List[Dict]
    required_exams: List[BilingualText]
    risk_level: BilingualText
    recommendation: BilingualText
    treatment_priority: int
    processing_time_ms: float
    timestamp: datetime
    raw_input_text: str
    detected_language: Language
    ai_model_version: str = "1.0.0"
    
    def to_create_dict(self) -> Dict:
        """Convert for database insertion"""
        return {
            "organization_id": self.organization_id,
            "session_id": self.session_id,
            "condition_icd10": self.condition_id,
            "condition_name_en": self.condition_name.en,
            "condition_name_pt": self.condition_name.pt,
            "confidence": self.confidence,
            "severity": self.severity.value,
            "urgency": self.urgency.value,
            "treatment_priority": self.treatment_priority,
            "processing_time_ms": int(self.processing_time_ms),
            "recommendation_en": self.recommendation.en,
            "recommendation_pt": self.recommendation.pt,
            "risk_level_en": self.risk_level.en,
            "risk_level_pt": self.risk_level.pt,
            "raw_input_text": self.raw_input_text,
            "detected_language": self.detected_language.value,
            "ai_model_version": self.ai_model_version
        }