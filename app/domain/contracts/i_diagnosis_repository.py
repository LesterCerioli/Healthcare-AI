"""Abstract contract for the diagnosis repository."""

from abc import ABC, abstractmethod
from typing import List, Optional

from app.domain.models.diagnosis import Diagnosis


class IDiagnosisRepository(ABC):
    """Defines the persistence operations for Diagnosis entities.

    Services depend on this interface, not on any concrete implementation,
    keeping the domain layer free of infrastructure concerns.
    """

    @abstractmethod
    async def create(self, diagnosis: Diagnosis) -> Diagnosis:
        """Persist a new Diagnosis and return it with its assigned id.

        Parameters
        ----------
        diagnosis:
            The diagnosis entity to persist.  Its ``id`` field may be None;
            the repository implementation assigns the actual primary key.

        Returns
        -------
        Diagnosis
            The persisted entity, including the database-assigned ``id`` and
            ``created_at`` timestamp.
        """

    @abstractmethod
    async def get_by_id(self, diagnosis_id: int) -> Optional[Diagnosis]:
        """Return the Diagnosis with the given primary key, or None.

        Parameters
        ----------
        diagnosis_id:
            The integer primary key to look up.
        """

    @abstractmethod
    async def get_by_patient_id(self, patient_id: int) -> List[Diagnosis]:
        """Return all diagnoses associated with the given patient.

        Parameters
        ----------
        patient_id:
            The patient's primary key.

        Returns
        -------
        List[Diagnosis]
            Ordered list (most recent first) of diagnoses for that patient.
            Returns an empty list if no records are found.
        """
