"""
Storage layer exception hierarchy for Phase 8A MongoDB Quarantine Store.
Project: Self-Adaptive Zero Trust Cybersecurity Architecture for Industry 5.0
"""

class StorageError(Exception):
    """Base exception for all storage-related errors."""
    pass


class DatabaseConnectionError(StorageError):
    """Raised when connecting to or communicating with MongoDB fails."""
    pass


class DuplicateIncidentError(StorageError):
    """Raised when attempting to insert an incident with an already existing incident_id."""
    pass


class IncidentNotFoundError(StorageError):
    """Raised when an incident is not found in the quarantine store."""
    pass


class InvalidStatusTransitionError(StorageError):
    """Raised when an invalid lifecycle status transition is attempted."""
    pass


class IneligibleQuarantineError(StorageError):
    """Raised when an incident action is not eligible for quarantine persistence."""
    pass
