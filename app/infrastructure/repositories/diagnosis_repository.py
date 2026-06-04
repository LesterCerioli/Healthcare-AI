"""Concrete SQLAlchemy implementation of IDiagnosisRepository."""

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import Column, DateTime, Float, Integer, JSON, String, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.contracts.i_diagnosis_repository import IDiagnosisRepository
from app.domain.models.diagnosis import Diagnosis
from app.infrastructure.database.connection import Base


# ---------------------------------------------------------------------------
# ORM table model
# ---------------------------------------------------------------------------

class DiagnosisRecord(Base):
    """SQLAlchemy ORM representation of the ``diagnoses`` table."""

    __tablename__ = "diagnoses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, nullable=False, index=True)
    primary_diagnosis = Column(String(255), nullable=False)
    confidence = Column(Float, nullable=False)
    urgency_level = Column(String(50), nullable=False)
    differential_diagnoses = Column(JSON, nullable=False, default=list)
    recommended_examinations = Column(JSON, nullable=False, default=list)
    treatment_suggestions = Column(JSON, nullable=False, default=list)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Repository implementation
# ---------------------------------------------------------------------------

class DiagnosisRepository(IDiagnosisRepository):
    """Async SQLAlchemy implementation of IDiagnosisRepository."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, diagnosis: Diagnosis) -> Diagnosis:
        record = DiagnosisRecord(
            patient_id=diagnosis.patient_id,
            primary_diagnosis=diagnosis.primary_diagnosis,
            confidence=diagnosis.confidence,
            urgency_level=diagnosis.urgency_level,
            differential_diagnoses=diagnosis.differential_diagnoses,
            recommended_examinations=diagnosis.recommended_examinations,
            treatment_suggestions=diagnosis.treatment_suggestions,
            created_at=diagnosis.created_at or datetime.now(timezone.utc),
        )
        self._session.add(record)
        await self._session.flush()
        await self._session.refresh(record)
        return self._map_to_domain(record)

    async def get_by_id(self, diagnosis_id: int) -> Optional[Diagnosis]:
        result = await self._session.execute(
            select(DiagnosisRecord).where(DiagnosisRecord.id == diagnosis_id)
        )
        record = result.scalar_one_or_none()
        return self._map_to_domain(record) if record else None

    async def get_by_patient_id(self, patient_id: int) -> List[Diagnosis]:
        result = await self._session.execute(
            select(DiagnosisRecord)
            .where(DiagnosisRecord.patient_id == patient_id)
            .order_by(DiagnosisRecord.created_at.desc())
        )
        records = result.scalars().all()
        return [self._map_to_domain(r) for r in records]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_to_domain(record: DiagnosisRecord) -> Diagnosis:
        return Diagnosis(
            id=record.id,
            patient_id=record.patient_id,
            primary_diagnosis=record.primary_diagnosis,
            confidence=record.confidence,
            urgency_level=record.urgency_level,
            differential_diagnoses=record.differential_diagnoses or [],
            recommended_examinations=record.recommended_examinations or [],
            treatment_suggestions=record.treatment_suggestions or [],
            created_at=record.created_at,
        )
