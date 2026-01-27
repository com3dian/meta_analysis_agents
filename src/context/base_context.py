"""
ExecutionContext Base Classes and Models.

This module defines the abstract base class for all execution contexts and
common data models used across the system.

The ExecutionContext abstraction provides a unified interface for the "world"
in which the agents operate. This can be:
- A structured data source (database, CSVs)
- A codebase (file system)
- An API
- A website
- etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd


class ContextType(str, Enum):
    """Enumeration of supported execution context types."""

    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    SQLITE = "sqlite"
    DIRECTORY = "directory"
    API = "api"
    WEBSITE = "website"
    UNKNOWN = "unknown"


@dataclass
class FieldInfo:
    """Information about a single field within a resource."""

    name: str
    dtype: str
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_reference: Optional[str] = None  # "resource.field" format
    description: Optional[str] = None
    sample_values: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "nullable": self.nullable,
            "is_primary_key": self.is_primary_key,
            "is_foreign_key": self.is_foreign_key,
            "foreign_key_reference": self.foreign_key_reference,
            "description": self.description,
            "sample_values": self.sample_values[:5],
        }


@dataclass
class ResourceInfo:
    """Information about a single resource in the execution context."""

    name: str
    item_count: Optional[int] = None  # e.g., row_count
    field_count: Optional[int] = None  # e.g., column_count
    fields: List[FieldInfo] = field(default_factory=list)
    primary_key: Optional[str] = None
    location: Optional[str] = None  # For file-based or URL-based resources
    size_in_bytes: Optional[int] = None
    description: Optional[str] = None

    @property
    def field_names(self) -> List[str]:
        return [f.name for f in self.fields]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "item_count": self.item_count,
            "field_count": self.field_count,
            "fields": [f.to_dict() for f in self.fields],
            "primary_key": self.primary_key,
            "location": self.location,
            "size_in_bytes": self.size_in_bytes,
            "description": self.description,
        }


@dataclass
class RelationshipInfo:
    """Information about a relationship between resources."""

    from_resource: str
    from_field: str
    to_resource: str
    to_field: str
    relationship_type: str  # "one-to-one", "one-to-many", "many-to-many"
    confidence: float = 0.0
    is_verified: bool = False
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_resource": self.from_resource,
            "from_field": self.from_field,
            "to_resource": self.to_resource,
            "to_field": self.to_field,
            "relationship_type": self.relationship_type,
            "confidence": self.confidence,
            "is_verified": self.is_verified,
            "description": self.description,
        }


class ExecutionContext(ABC):
    """
    Abstract base class for all execution contexts.
    Provides a unified interface for accessing the operational environment.
    """

    def __init__(self, name: str = "context", description: Optional[str] = None):
        self._name = name
        self._description = description
        self._resource_cache: Dict[str, ResourceInfo] = {}
        self._relationship_cache: Optional[List[RelationshipInfo]] = None

    @property
    @abstractmethod
    def context_type(self) -> ContextType:
        """Return the type of this context."""
        pass

    @property
    @abstractmethod
    def resources(self) -> List[str]:
        """Return list of resource names in this context."""
        pass

    @abstractmethod
    def _load_resource_info(self, resource: str) -> ResourceInfo:
        """Load metadata for a specific resource."""
        pass

    @abstractmethod
    def read_resource(
        self,
        resource: str,
        fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Read a resource into a pandas DataFrame."""
        pass

    @abstractmethod
    def iter_resource(
        self, resource: str, chunksize: int = 10000, **kwargs
    ) -> Iterator[pd.DataFrame]:
        """Iterate over a resource in chunks."""
        pass

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> Optional[str]:
        return self._description

    @property
    def is_multi_resource(self) -> bool:
        return len(self.resources) > 1

    @property
    def primary_resource(self) -> Optional[str]:
        return self.resources[0] if self.resources else None

    def get_resource_info(self, resource: str) -> ResourceInfo:
        if resource not in self.resources:
            raise ValueError(
                f"Resource '{resource}' not found. Available: {self.resources}"
            )

        if resource not in self._resource_cache:
            self._resource_cache[resource] = self._load_resource_info(resource)

        return self._resource_cache[resource]

    def get_all_resource_info(self) -> Dict[str, ResourceInfo]:
        return {
            resource: self.get_resource_info(resource) for resource in self.resources
        }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "context_type": self.context_type.value,
            "is_multi_resource": self.is_multi_resource,
            "resources": {
                name: info.to_dict()
                for name, info in self.get_all_resource_info().items()
            },
            "relationships": [r.to_dict() for r in self.get_relationships()],
        }

    def get_relationships(self) -> List[RelationshipInfo]:
        if self._relationship_cache is None:
            self._relationship_cache = self._discover_relationships()
        return self._relationship_cache

    def _discover_relationships(self) -> List[RelationshipInfo]:
        return []

    def get_field_values(
        self, resource: str, field: str, limit: Optional[int] = None
    ) -> List[Any]:
        df = self.read_resource(resource, fields=[field])
        values = df[field].dropna().unique().tolist()
        if limit:
            return values[:limit]
        return values

    def validate(self) -> bool:
        if not self.resources:
            raise ValueError("Context has no resources")

        for resource in self.resources:
            self.get_resource_info(resource)

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "context_type": self.context_type.value,
            "resources": self.resources,
            "is_multi_resource": self.is_multi_resource,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"resources={self.resources}, "
            f"context_type='{self.context_type.value}')"
        )

    def __str__(self) -> str:
        resource_info = (
            f"{len(self.resources)} resource(s)"
            if self.is_multi_resource
            else self.resources[0]
        )
        return f"{self.name} ({self.context_type.value}: {resource_info})"
