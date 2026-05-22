
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import json

from app.database import async_db
from app.models import DiagnosticResult, OrganizationSettings

class DiagnosticRepository:
        
    @staticmethod
    async def validate_organization(organization_id: uuid.UUID) -> bool:
        
        async with async_db.get_connection() as conn:
            row = await conn.fetchrow("""
                SELECT id FROM public.organizations 
                WHERE id = $1 AND is_active = TRUE
            """, organization_id)
            return row is not None
    
    @staticmethod
    async def create_or_get_session(
        organization_id: uuid.UUID, 
        session_identifier: str
    ) -> uuid.UUID:
                
        async with async_db.get_connection() as conn:
            
            row = await conn.fetchrow("""
                SELECT id FROM public.diagnostic_sessions 
                WHERE organization_id = $1 AND session_identifier = $2
            """, organization_id, session_identifier)
            
            if row:
                return row['id']
            
            
            row = await conn.fetchrow("""
                INSERT INTO public.diagnostic_sessions (organization_id, session_identifier)
                VALUES ($1, $2)
                RETURNING id
            """, organization_id, session_identifier)
                        
            await conn.execute("""
                INSERT INTO public.diagnostic_audit_log (
                    organization_id, action, resource_type, resource_id, details
                ) VALUES ($1, $2, $3, $4, $5)
            """, organization_id, "SESSION_CREATED", "diagnostic_sessions", row['id'], 
                json.dumps({"session_identifier": session_identifier}))
            
            return row['id']
    
    @staticmethod
    async def save_diagnostic_result(result: DiagnosticResult) -> uuid.UUID:
                
        create_dict = result.to_create_dict()
        
        async with async_db.get_connection() as conn:
            
            row = await conn.fetchrow("""
                INSERT INTO public.diagnostic_results (
                    organization_id, session_id, condition_icd10, 
                    condition_name_en, condition_name_pt, confidence, 
                    severity, urgency, treatment_priority, processing_time_ms,
                    recommendation_en, recommendation_pt, risk_level_en, 
                    risk_level_pt, raw_input_text, detected_language, ai_model_version
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                RETURNING id
            """,
                create_dict["organization_id"],
                create_dict["session_id"],
                create_dict["condition_icd10"],
                create_dict["condition_name_en"],
                create_dict["condition_name_pt"],
                create_dict["confidence"],
                create_dict["severity"],
                create_dict["urgency"],
                create_dict["treatment_priority"],
                create_dict["processing_time_ms"],
                create_dict["recommendation_en"],
                create_dict["recommendation_pt"],
                create_dict["risk_level_en"],
                create_dict["risk_level_pt"],
                create_dict["raw_input_text"],
                create_dict["detected_language"],
                create_dict["ai_model_version"]
            )
            
            diagnostic_result_id = row['id']
                        
            for symptom in result.matched_symptoms:
                await conn.execute("""
                    INSERT INTO public.matched_symptoms (
                        organization_id, diagnostic_result_id, symptom_id, 
                        symptom_name_en, symptom_name_pt, confidence, severity_weight
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                    result.organization_id,
                    diagnostic_result_id,
                    symptom.get('id', 'unknown'),
                    symptom.get('name_en', ''),
                    symptom.get('name_pt', ''),
                    symptom.get('confidence', 0.0),
                    symptom.get('severity_weight', 0.0)
                )
            
            
            for diff in result.differential_diagnoses:
                await conn.execute("""
                    INSERT INTO public.differential_diagnoses (
                        organization_id, diagnostic_result_id, condition_icd10,
                        condition_name_en, condition_name_pt, confidence
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """,
                    result.organization_id,
                    diagnostic_result_id,
                    diff.get('icd10', ''),
                    diff.get('name_en', ''),
                    diff.get('name_pt', ''),
                    diff.get('confidence', 0.0)
                )
            
            
            for exam in result.required_exams:
                await conn.execute("""
                    INSERT INTO public.required_exams (
                        organization_id, diagnostic_result_id, 
                        exam_name_en, exam_name_pt
                    ) VALUES ($1, $2, $3, $4)
                """,
                    result.organization_id,
                    diagnostic_result_id,
                    exam.en,
                    exam.pt
                )
            
            
            await conn.execute("""
                INSERT INTO public.diagnostic_audit_log (
                    organization_id, action, resource_type, resource_id, details
                ) VALUES ($1, $2, $3, $4, $5)
            """, result.organization_id, "DIAGNOSIS_CREATED", "diagnostic_results", 
                diagnostic_result_id, json.dumps({"condition": result.condition_id, "confidence": result.confidence}))
            
            return diagnostic_result_id
    
    @staticmethod
    async def get_organization_settings(organization_id: uuid.UUID) -> OrganizationSettings:
                
        async with async_db.get_connection() as conn:
            row = await conn.fetchrow("""
                SELECT 
                    id as organization_id,
                    settings->>'preferred_language' as preferred_language,
                    (settings->>'custom_treatments_enabled')::boolean as custom_treatments_enabled,
                    (settings->>'ai_confidence_threshold')::float as ai_confidence_threshold,
                    (settings->>'audit_enabled')::boolean as audit_enabled
                FROM public.organizations
                WHERE id = $1
            """, organization_id)
            
            if row:
                return OrganizationSettings(
                    organization_id=row['organization_id'],
                    custom_treatments_enabled=row.get('custom_treatments_enabled', False),
                    preferred_language=Language(row.get('preferred_language', 'en')),
                    ai_confidence_threshold=row.get('ai_confidence_threshold', 0.65),
                    audit_enabled=row.get('audit_enabled', True)
                )
            
            
            return OrganizationSettings(organization_id=organization_id)
    
    @staticmethod
    async def get_custom_treatment(
        organization_id: uuid.UUID, 
        condition_icd10: str
    ) -> Optional[str]:
                
        async with async_db.get_connection() as conn:
            row = await conn.fetchrow("""
                SELECT custom_treatment_en, custom_treatment_pt
                FROM public.organization_treatment_protocols
                WHERE organization_id = $1 AND condition_icd10 = $2 AND is_active = TRUE
            """, organization_id, condition_icd10)
            
            if row:
                return {
                    "en": row['custom_treatment_en'],
                    "pt": row['custom_treatment_pt']
                }
            return None
    
    @staticmethod
    async def get_diagnostic_history(
        organization_id: uuid.UUID, 
        session_id: uuid.UUID, 
        limit: int = 10
    ) -> List[Dict]:
                
        async with async_db.get_connection() as conn:
            rows = await conn.fetch("""
                SELECT 
                    id,
                    condition_icd10,
                    condition_name_en,
                    condition_name_pt,
                    confidence,
                    severity,
                    urgency,
                    recommendation_en,
                    recommendation_pt,
                    created_at
                FROM public.diagnostic_results
                WHERE organization_id = $1 AND session_id = $2
                ORDER BY created_at DESC
                LIMIT $3
            """, organization_id, session_id, limit)
            
            return [dict(row) for row in rows]
    
    @staticmethod
    async def get_organization_statistics(organization_id: uuid.UUID) -> Dict:
                
        async with async_db.get_connection() as conn:
            
            await conn.execute(f"SET app.current_organization_id = '{organization_id}'")
                        
            totals = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_diagnoses,
                    COUNT(DISTINCT session_id) as unique_sessions,
                    AVG(confidence) as avg_confidence,
                    AVG(processing_time_ms) as avg_processing_time_ms,
                    MIN(created_at) as first_diagnosis,
                    MAX(created_at) as last_diagnosis
                FROM public.diagnostic_results
            """)
            
            
            severity_counts = await conn.fetch("""
                SELECT 
                    severity,
                    COUNT(*) as count
                FROM public.diagnostic_results
                GROUP BY severity
                ORDER BY severity
            """)
            
            
            common_diagnoses = await conn.fetch("""
                SELECT 
                    condition_icd10,
                    condition_name_en,
                    COUNT(*) as count,
                    AVG(confidence) as avg_confidence
                FROM public.diagnostic_results
                GROUP BY condition_icd10, condition_name_en
                ORDER BY count DESC
                LIMIT 10
            """)
            
            
            daily_trend = await conn.fetch("""
                SELECT 
                    DATE(created_at) as date,
                    COUNT(*) as diagnoses_count
                FROM public.diagnostic_results
                WHERE created_at >= NOW() - INTERVAL '30 days'
                GROUP BY DATE(created_at)
                ORDER BY date DESC
            """)
            
            return {
                "organization_id": str(organization_id),
                "total_diagnoses": totals['total_diagnoses'] if totals else 0,
                "unique_sessions": totals['unique_sessions'] if totals else 0,
                "avg_confidence": float(totals['avg_confidence']) if totals and totals['avg_confidence'] else 0,
                "avg_processing_time_ms": float(totals['avg_processing_time_ms']) if totals and totals['avg_processing_time_ms'] else 0,
                "first_diagnosis": totals['first_diagnosis'].isoformat() if totals and totals['first_diagnosis'] else None,
                "last_diagnosis": totals['last_diagnosis'].isoformat() if totals and totals['last_diagnosis'] else None,
                "severity_breakdown": [dict(row) for row in severity_counts],
                "top_diagnoses": [dict(row) for row in common_diagnoses],
                "daily_trend": [dict(row) for row in daily_trend]
            }
repository = Repository()