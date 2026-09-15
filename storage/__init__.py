"""
Storage layer package exports for Phase 8A MongoDB Quarantine Store.
Project: Self-Adaptive Zero Trust Cybersecurity Architecture for Industry 5.0
"""

from storage.config import MongoConfig, get_mongo_config
from storage.exceptions import (
    DatabaseConnectionError,
    DuplicateIncidentError,
    IncidentNotFoundError,
    IneligibleQuarantineError,
    InvalidStatusTransitionError,
    StorageError,
)
from storage.mongodb import MongoDBManager
from storage.quarantine_store import (
    BaseQuarantineStore,
    InMemoryQuarantineStore,
    MongoQuarantineStore,
    create_quarantine_store,
)
from storage.schemas import (
    ALLOWED_TRANSITIONS,
    QUARANTINE_ELIGIBLE_ACTIONS,
    IncidentStatus,
    QuarantineIncident,
    StatusHistoryEntry,
    is_quarantine_eligible,
    quarantine_incident_from_phase7_state,
    validate_status_transition,
)

__all__ = [
    "MongoConfig",
    "get_mongo_config",
    "StorageError",
    "DatabaseConnectionError",
    "DuplicateIncidentError",
    "IncidentNotFoundError",
    "InvalidStatusTransitionError",
    "IneligibleQuarantineError",
    "MongoDBManager",
    "BaseQuarantineStore",
    "MongoQuarantineStore",
    "InMemoryQuarantineStore",
    "create_quarantine_store",
    "IncidentStatus",
    "ALLOWED_TRANSITIONS",
    "QUARANTINE_ELIGIBLE_ACTIONS",
    "QuarantineIncident",
    "StatusHistoryEntry",
    "is_quarantine_eligible",
    "quarantine_incident_from_phase7_state",
    "validate_status_transition",
]
