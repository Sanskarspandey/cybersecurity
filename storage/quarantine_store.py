"""
Quarantine Store repository implementation for Phase 8A.
Project: Self-Adaptive Zero Trust Cybersecurity Architecture for Industry 5.0

Provides an abstracted repository for storing, retrieving, and managing the lifecycle
of quarantined security incidents with strict schema validation and idempotency guarantees.
Supports live MongoDB storage as well as an in-memory test store.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import pymongo
from pymongo.errors import DuplicateKeyError

from storage.config import MongoConfig, get_mongo_config
from storage.exceptions import (
    DatabaseConnectionError,
    DuplicateIncidentError,
    IncidentNotFoundError,
    InvalidStatusTransitionError,
)
from storage.mongodb import MongoDBManager
from storage.schemas import (
    IncidentStatus,
    QuarantineIncident,
    StatusHistoryEntry,
    validate_status_transition,
)


class BaseQuarantineStore:
    """Abstract interface defining required quarantine store operations."""

    def health_check(self) -> bool:
        raise NotImplementedError

    def insert_quarantine_incident(self, incident: QuarantineIncident) -> str:
        raise NotImplementedError

    def get_incident_by_id(self, incident_id: str) -> Optional[QuarantineIncident]:
        raise NotImplementedError

    def list_quarantine_incidents(
        self,
        status: Optional[Union[IncidentStatus, str]] = None,
        limit: int = 100,
    ) -> List[QuarantineIncident]:
        raise NotImplementedError

    def update_incident_status(
        self,
        incident_id: str,
        new_status: Union[IncidentStatus, str],
        comment: Optional[str] = None,
        updated_by: Optional[str] = "analyst",
    ) -> QuarantineIncident:
        raise NotImplementedError


class MongoQuarantineStore(BaseQuarantineStore):
    """
    MongoDB-backed implementation of the Quarantine Store.
    Enforces unique incident_id indexing, idempotency, and status lifecycle transitions.
    """

    def __init__(self, manager: Optional[MongoDBManager] = None):
        self.manager = manager or MongoDBManager()
        self._initialized = False

    def _ensure_ready(self) -> None:
        """Connects and configures collection indexes if not already initialized."""
        if not self._initialized:
            self.manager.ensure_indexes()
            self._initialized = True

    def health_check(self) -> bool:
        """Checks whether the MongoDB server is reachable."""
        return self.manager.health_check()

    def insert_quarantine_incident(self, incident: QuarantineIncident) -> str:
        """
        Inserts a validated QuarantineIncident document into MongoDB.
        Raises DuplicateIncidentError if an incident with the same incident_id exists.
        """
        self._ensure_ready()
        col = self.manager.get_collection()

        doc = incident.model_dump(mode="json")
        try:
            col.insert_one(doc)
            return incident.incident_id
        except DuplicateKeyError as e:
            raise DuplicateIncidentError(
                f"Quarantine incident '{incident.incident_id}' already exists in store."
            ) from e

    def get_incident_by_id(self, incident_id: str) -> Optional[QuarantineIncident]:
        """
        Retrieves a quarantine incident by unique incident_id.
        Returns None if not found.
        """
        self._ensure_ready()
        col = self.manager.get_collection()
        doc = col.find_one({"incident_id": incident_id}, {"_id": 0})
        if not doc:
            return None
        return QuarantineIncident(**doc)

    def list_quarantine_incidents(
        self,
        status: Optional[Union[IncidentStatus, str]] = None,
        limit: int = 100,
    ) -> List[QuarantineIncident]:
        """
        Retrieves recent quarantine incidents, optionally filtered by status.
        Ordered by created_at descending.
        """
        self._ensure_ready()
        col = self.manager.get_collection()

        query: Dict[str, Any] = {}
        if status is not None:
            st = IncidentStatus(status) if isinstance(status, str) else status
            query["incident_status"] = st.value

        cursor = col.find(query, {"_id": 0}).sort("created_at", pymongo.DESCENDING).limit(limit)
        return [QuarantineIncident(**doc) for doc in cursor]

    def update_incident_status(
        self,
        incident_id: str,
        new_status: Union[IncidentStatus, str],
        comment: Optional[str] = None,
        updated_by: Optional[str] = "analyst",
    ) -> QuarantineIncident:
        """
        Executes a controlled status transition for an existing quarantine incident.
        Validates the transition against the lifecycle state machine and appends an audit record.
        """
        self._ensure_ready()
        target_status = IncidentStatus(new_status) if isinstance(new_status, str) else new_status

        # 1. Retrieve current document
        current = self.get_incident_by_id(incident_id)
        if current is None:
            raise IncidentNotFoundError(f"Quarantine incident '{incident_id}' not found in store.")

        # 2. Validate state transition
        validate_status_transition(current.incident_status, target_status)

        # 3. Prepare update document
        now_iso = datetime.now(timezone.utc).isoformat()
        history_entry = StatusHistoryEntry(
            from_status=current.incident_status.value,
            to_status=target_status.value,
            timestamp=now_iso,
            updated_by=updated_by or "analyst",
            comment=comment,
        )

        col = self.manager.get_collection()
        col.update_one(
            {"incident_id": incident_id},
            {
                "$set": {
                    "incident_status": target_status.value,
                    "updated_at": now_iso,
                },
                "$push": {
                    "status_history": history_entry.model_dump(mode="json"),
                },
            },
        )

        # 4. Return refreshed document
        updated_doc = self.get_incident_by_id(incident_id)
        assert updated_doc is not None
        return updated_doc


class InMemoryQuarantineStore(BaseQuarantineStore):
    """
    In-memory mock store implementing the exact BaseQuarantineStore interface
    for isolated, hermetic unit tests when MongoDB is not running.
    """

    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}

    def health_check(self) -> bool:
        return True

    def insert_quarantine_incident(self, incident: QuarantineIncident) -> str:
        if incident.incident_id in self._data:
            raise DuplicateIncidentError(
                f"Quarantine incident '{incident.incident_id}' already exists in store."
            )
        self._data[incident.incident_id] = incident.model_dump(mode="json")
        return incident.incident_id

    def get_incident_by_id(self, incident_id: str) -> Optional[QuarantineIncident]:
        doc = self._data.get(incident_id)
        if not doc:
            return None
        return QuarantineIncident(**doc)

    def list_quarantine_incidents(
        self,
        status: Optional[Union[IncidentStatus, str]] = None,
        limit: int = 100,
    ) -> List[QuarantineIncident]:
        results = []
        target = IncidentStatus(status).value if status else None
        for doc in self._data.values():
            if target is None or doc.get("incident_status") == target:
                results.append(QuarantineIncident(**doc))
        # Sort by created_at descending
        results.sort(key=lambda x: x.created_at, reverse=True)
        return results[:limit]

    def update_incident_status(
        self,
        incident_id: str,
        new_status: Union[IncidentStatus, str],
        comment: Optional[str] = None,
        updated_by: Optional[str] = "analyst",
    ) -> QuarantineIncident:
        current = self.get_incident_by_id(incident_id)
        if current is None:
            raise IncidentNotFoundError(f"Quarantine incident '{incident_id}' not found in store.")

        target = IncidentStatus(new_status) if isinstance(new_status, str) else new_status
        validate_status_transition(current.incident_status, target)

        now_iso = datetime.now(timezone.utc).isoformat()
        history_entry = StatusHistoryEntry(
            from_status=current.incident_status.value,
            to_status=target.value,
            timestamp=now_iso,
            updated_by=updated_by or "analyst",
            comment=comment,
        )

        doc = self._data[incident_id]
        doc["incident_status"] = target.value
        doc["updated_at"] = now_iso
        doc.setdefault("status_history", []).append(history_entry.model_dump(mode="json"))

        return QuarantineIncident(**doc)


def create_quarantine_store(use_in_memory: bool = False) -> BaseQuarantineStore:
    """
    Factory function instantiating either the live MongoQuarantineStore
    or the hermetic InMemoryQuarantineStore.
    """
    if use_in_memory:
        return InMemoryQuarantineStore()
    return MongoQuarantineStore()
