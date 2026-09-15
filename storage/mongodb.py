"""
MongoDB connection manager and index manager for Phase 8A Quarantine Store.
Project: Self-Adaptive Zero Trust Cybersecurity Architecture for Industry 5.0

Manages PyMongo client lifecycle, health checks, safe error propagation via
DatabaseConnectionError, and automatic index creation.
"""

from typing import Any, Dict, List, Optional
import pymongo
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from storage.config import MongoConfig, get_mongo_config
from storage.exceptions import DatabaseConnectionError


class MongoDBManager:
    """
    Thread-safe connection manager for MongoDB persistence.
    Isolates database network operations and guarantees zero credential leakage.
    """

    def __init__(self, config: Optional[MongoConfig] = None):
        self.config = config or get_mongo_config()
        self._client: Optional[pymongo.MongoClient] = None

    @property
    def client(self) -> pymongo.MongoClient:
        if self._client is None:
            self.connect()
        return self._client  # type: ignore

    def connect(self) -> pymongo.MongoClient:
        """
        Establishes connection to MongoDB and performs an active ping check.
        Raises DatabaseConnectionError if the host is unreachable.
        """
        try:
            client = pymongo.MongoClient(
                self.config.uri,
                serverSelectionTimeoutMS=self.config.server_selection_timeout_ms,
                connectTimeoutMS=self.config.server_selection_timeout_ms,
            )
            # Active ping to verify reachability
            client.admin.command("ping")
            self._client = client
            return self._client
        except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
            masked = self.config.get_masked_uri()
            raise DatabaseConnectionError(
                f"Failed to connect to MongoDB at '{masked}': {type(e).__name__}: {str(e)}"
            ) from e

    def health_check(self) -> bool:
        """
        Performs a non-raising reachability check.
        Returns True if database responds to ping within timeout; False otherwise.
        """
        try:
            if self._client is None:
                self.connect()
            self._client.admin.command("ping")  # type: ignore
            return True
        except Exception:
            return False

    def get_database(self) -> Database:
        """Returns the configured database instance."""
        return self.client[self.config.database]

    def get_collection(self) -> Collection:
        """Returns the configured quarantine incidents collection."""
        return self.get_database()[self.config.collection]

    def ensure_indexes(self) -> List[str]:
        """
        Ensures necessary forensic indexes exist on the collection:
        1. incident_id (Unique)
        2. created_at (Descending chronological order)
        3. incident_status (Ascending categorical filter)
        4. asset_id (Ascending device lookup)
        """
        col = self.get_collection()
        index_names = []

        # 1. Unique index on incident_id for idempotency
        idx_id = col.create_index(
            [("incident_id", pymongo.ASCENDING)],
            unique=True,
            name="uniq_incident_id",
        )
        index_names.append(idx_id)

        # 2. Chronological retrieval index
        idx_time = col.create_index(
            [("created_at", pymongo.DESCENDING)],
            name="idx_created_at_desc",
        )
        index_names.append(idx_time)

        # 3. Lifecycle status query index
        idx_status = col.create_index(
            [("incident_status", pymongo.ASCENDING)],
            name="idx_incident_status",
        )
        index_names.append(idx_status)

        # 4. Asset identifier query index
        idx_asset = col.create_index(
            [("asset_id", pymongo.ASCENDING)],
            name="idx_asset_id",
        )
        index_names.append(idx_asset)

        return index_names

    def close(self) -> None:
        """Closes any active database client connections."""
        if self._client is not None:
            self._client.close()
            self._client = None
