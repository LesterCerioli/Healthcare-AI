
from fastapi import FastAPI, HTTPException, Depends, Header
from typing import Optional
import uuid
from datetime import datetime

from app.database import async_db
from app.models import DiagnosisRequest, DiagnosisResponse, Language, DiagnosticResult, BilingualText
from app.diagnosis_engine import AsyncDiagnosticEngine
from app.repository import DiagnosticRepository
from app.treatment_engine import AsyncTreatmentEngine

app = FastAPI(title="Medical Diagnosis SaaS API", version="1.0.0")


diagnostic_engine = AsyncDiagnosticEngine()
treatment_engine = AsyncTreatmentEngine()

async def get_organization(
    organization_id: uuid.UUID = Header(..., alias="X-Organization-ID")
) -> uuid.UUID:
    """Extract and validate organization ID from header"""
    if not await DiagnosticRepository.validate_organization(organization_id):
        raise HTTPException(
            status_code=403, 
            detail="Invalid or inactive organization ID"
        )
    return organization_id

@app.on_event("startup")
async def startup():
    await async_db.connect()
    await diagnostic_engine.initialize()
    await treatment_engine.initialize()
    print("Medical Diagnosis SaaS API started")

@app.on_event("shutdown")
async def shutdown():
    await async_db.close()
    print("🛑 Medical Diagnosis SaaS API stopped")

@app.post("/api/diagnose", response_model=DiagnosisResponse)
async def diagnose(
    request: DiagnosisRequest,
    organization_id: uuid.UUID = Depends(get_organization)
):
    """
    Complete diagnosis endpoint with multi-tenant isolation.
    Organization ID must be provided in X-Organization-ID header.
    """
    
    
    session_identifier = request.session_identifier or str(uuid.uuid4())
        
    session_id = await DiagnosticRepository.create_or_get_session(
        organization_id, 
        session_identifier
    )
        
    org_settings = await DiagnosticRepository.get_organization_settings(organization_id)
        
    language = request.language or org_settings.preferred_language
        
    start_time = datetime.now()
        
    diagnosis = await diagnostic_engine.diagnose(
        request.text, 
        language,
        patient_context={
            "age": request.patient_age,
            "gender": request.patient_gender,
            "known_conditions": request.known_conditions,
            "organization_id": str(organization_id)
        }
    )
    
    
    custom_treatment = await DiagnosticRepository.get_custom_treatment(
        organization_id, 
        diagnosis.condition_id
    )
        
    if custom_treatment and org_settings.custom_treatments_enabled:
        diagnosis.recommendation = BilingualText(
            en=custom_treatment['en'],
            pt=custom_treatment['pt']
        )
    
    processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
    
    
    diagnostic_result = DiagnosticResult(
        organization_id=organization_id,
        session_id=session_id,
        condition_id=diagnosis.condition_id,
        condition_name=diagnosis.condition_name,
        confidence=diagnosis.confidence,
        severity=diagnosis.severity,
        urgency=diagnosis.urgency,
        matched_symptoms=diagnosis.matched_symptoms,
        differential_diagnoses=diagnosis.differential_diagnoses,
        required_exams=diagnosis.required_exams,
        risk_level=diagnosis.risk_level,
        recommendation=diagnosis.recommendation,
        treatment_priority=diagnosis.treatment_priority,
        processing_time_ms=processing_time_ms,
        timestamp=datetime.now(),
        raw_input_text=request.text,
        detected_language=language,
        ai_model_version="1.0.0"
    )
    
    
    diagnostic_result_id = await DiagnosticRepository.save_diagnostic_result(diagnostic_result)
    
    return DiagnosisResponse(
        session_id=session_id,
        session_identifier=session_identifier,
        diagnostic_result_id=diagnostic_result_id,
        organization_id=organization_id,
        diagnosis=diagnosis.condition_name.get(language),
        confidence=diagnosis.confidence,
        severity=diagnosis.severity.value,
        urgency=diagnosis.urgency.value,
        recommendation=diagnosis.recommendation.get(language),
        treatment_priority=diagnosis.treatment_priority,
        processing_time_ms=processing_time_ms,
        created_at=datetime.now()
    )

@app.get("/api/organization/stats")
async def get_organization_stats(
    organization_id: uuid.UUID = Depends(get_organization)
):
    """Get diagnostic statistics for the organization"""
    return await DiagnosticRepository.get_organization_statistics(organization_id)

@app.get("/api/session/history/{session_identifier}")
async def get_session_history(
    session_identifier: str,
    organization_id: uuid.UUID = Depends(get_organization),
    limit: int = 10
):
    """Get diagnostic history for a session within organization"""
    
    
    async with async_db.get_connection() as conn:
        row = await conn.fetchrow("""
            SELECT id FROM diagnostic_sessions
            WHERE organization_id = $1 AND session_identifier = $2
        """, organization_id, session_identifier)
        
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_id = row['id']
    
    history = await DiagnosticRepository.get_diagnostic_history(
        organization_id, 
        session_id, 
        limit
    )
    
    return {
        "session_identifier": session_identifier,
        "session_id": session_id,
        "history": history
    }