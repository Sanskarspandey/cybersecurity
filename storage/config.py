"""
Configuration management for MongoDB Quarantine Store.
Project: Self-Adaptive Zero Trust Cybersecurity Architecture for Industry 5.0

Retrieves settings from environment variables with safe defaults and provides
strict credential masking to guarantee zero secret leakage in logs.
"""

import os
import re
from typing import NamedTuple


class MongoConfig(NamedTuple):
    """Immutable MongoDB connection configuration."""
    uri: str
    database: str
    collection: str
    server_selection_timeout_ms: int

    def get_masked_uri(self) -> str:
        """
        Returns the MongoDB URI with any username/password credentials sanitized.
        Example: mongodb://user:pass@localhost:27017 -> mongodb://****:****@localhost:27017
        """
        # Match standard connection string credentials: mongodb://[username:password@]host...
        pattern = r"://([^:]+):([^@]+)@"
        if re.search(pattern, self.uri):
            return re.sub(pattern, r"://****:****@", self.uri)
        return self.uri


def get_mongo_config() -> MongoConfig:
    """
    Constructs MongoConfig from environment variables with safe local defaults.
    """
    uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    database = os.environ.get("MONGODB_DATABASE", "industry5_zero_trust")
    collection = os.environ.get("MONGODB_QUARANTINE_COLLECTION", "quarantine_incidents")
    
    timeout_raw = os.environ.get("MONGODB_SERVER_TIMEOUT_MS", "2000")
    try:
        timeout_ms = int(timeout_raw)
    except ValueError:
        timeout_ms = 2000

    return MongoConfig(
        uri=uri,
        database=database,
        collection=collection,
        server_selection_timeout_ms=timeout_ms,
    )
