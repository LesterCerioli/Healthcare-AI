"""Concrete SQLAlchemy implementation of IPatientRepository."""

from datetime import date
from typing import List, Optional

from sqlalchemy import Column, Date, Integer, String, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.contracts.i_patient_repository import IPatientRepository
from app.domain.models.patient import Patient
from app.infrastructure.database.connection import Base


# ---------------------------------------------------------------------------
# ORM table model
# ---------------------------------------------------------------------------

class PatientRecord(Base):
    """SQLAlchemy ORM representation of the ``patients`` table."""

    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    date_of_birth = Column(Date, nullable=False)
    gender = Column(String(10), nullable=False)
    contact_phone = Column(String(20), nullable=True)
    organization_name = Column(String(255), nullable=True, index=True)


# ---------------------------------------------------------------------------
# Repository implementation
# ---------------------------------------------------------------------------

class PatientRepository(IPatientRepository):
    """Async SQLAlchemy implementation of IPatientRepository."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, patient: Patient) -> Patient:
        record = PatientRecord(
            name=patient.name,
            date_of_birth=patient.date_of_birth,
            gender=patient.gender,
            contact_phone=patient.contact_phone,
            organization_name=patient.organization_name,
        )
        self._session.add(record)
        await self._session.flush()
        await self._session.refresh(record)
        return self._map_to_domain(record)

    async def get_by_id(self, patient_id: int) -> Optional[Patient]:
        result = await self._session.execute(
            select(PatientRecord).where(PatientRecord.id == patient_id)
        )
        record = result.scalar_one_or_none()
        return self._map_to_domain(record) if record else None

    async def get_by_organization(self, organization_name: str) -> List[Patient]:
        result = await self._session.execute(
            select(PatientRecord).where(
                PatientRecord.organization_name == organization_name
            )
        )
        records = result.scalars().all()
        return [self._map_to_domain(r) for r in records]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_to_domain(record: PatientRecord) -> Patient:
        return Patient(
            id=record.id,
            name=record.name,
            date_of_birth=record.date_of_birth,
            gender=record.gender,
            contact_phone=record.contact_phone,
            organization_name=record.organization_name,
        )
