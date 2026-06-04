"""Abstract contract for the patient repository."""

from abc import ABC, abstractmethod
from typing import List, Optional

from app.domain.models.patient import Patient


class IPatientRepository(ABC):
    """Defines the persistence operations for Patient entities.

    Services depend on this interface, not on any concrete implementation.
    """

    @abstractmethod
    async def create(self, patient: Patient) -> Patient:
        """Persist a new Patient and return it with its assigned id.

        Parameters
        ----------
        patient:
            The patient entity to persist.  The ``id`` field may be None.

        Returns
        -------
        Patient
            The persisted entity with the database-assigned primary key.
        """

    @abstractmethod
    async def get_by_id(self, patient_id: int) -> Optional[Patient]:
        """Return the Patient with the given primary key, or None.

        Parameters
        ----------
        patient_id:
            The integer primary key to look up.
        """

    @abstractmethod
    async def get_by_organization(self, organization_name: str) -> List[Patient]:
        """Return all patients belonging to the given organization.

        Parameters
        ----------
        organization_name:
            The organization / hospital name to filter by.

        Returns
        -------
        List[Patient]
            All matching patients.  Returns an empty list when none are found.
        """
