"""
Schema Factory - Dynamically generates Pydantic models from metadata standard definitions.

This module provides a factory class that reads nested dict/JSON structures 
and generates Pydantic classes for structured output validation.

Example usage:
    from src.core.schema_factory import SchemaFactory
    from src.standards import METADATA_STANDARDS
    
    # Create schema from a standard
    factory = SchemaFactory()
    OutputSchema = factory.create_from_standard(
        METADATA_STANDARDS["climate_vs_cropyield"],
        record_class_name="YieldRecord",
        output_class_name="MetaAnalysisOutput",
        records_key="records"
    )
    
    # Use with orchestrator
    result = orchestrator.execute_plan(
        plan=plan,
        context=context,
        objective=objective,
        output_schema=OutputSchema
    )
"""

import json
import re
from typing import Annotated, Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, BeforeValidator, Field, create_model, model_validator


def _expand_wide_column_dict_to_records(
    data: Dict[str, Any],
    records_key: str,
) -> Optional[Dict[str, Any]]:
    """
    Convert a single JSON object with *parallel list columns* into row records.

    Some LLMs return one object like::

        {"Year": "2000", "yield_a": [1,2], "yield_b": [3,4]}

    instead of::

        {"yield_records": [{"Year":"2000","yield_a":1,"yield_b":3}, ...]}

    Pydantic would otherwise ignore unknown top-level keys and leave ``records_key`` empty.
    """
    existing = data.get(records_key)
    if isinstance(existing, list) and len(existing) > 0:
        return None

    list_lengths: List[int] = []
    for k, v in data.items():
        if k == records_key:
            continue
        if isinstance(v, list):
            if len(v) == 0:
                return None
            list_lengths.append(len(v))

    if not list_lengths:
        return None

    sizes = set(list_lengths)
    if len(sizes) != 1:
        return None

    n = list_lengths[0]
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        row: Dict[str, Any] = {}
        for k, v in data.items():
            if k == records_key:
                continue
            if isinstance(v, list):
                row[k] = v[i]
            else:
                row[k] = v
        rows.append(row)

    return {records_key: rows}


def _normalize_output_root_dict(data: Any, records_key: str) -> Any:
    """
    Fix common LLM JSON shapes before validating extraction wrapper models.

    Pydantic drops unknown top-level keys (``extra`` is ignored by default). If the
    model expects ``yield_records`` but the API returns ``{"records": [...]}``,
    validation yields an empty default list — silent zero records. This normalizer:

    - Wraps a bare JSON array as ``{records_key: array}``
    - Copies the first non-empty list from common alternate keys into ``records_key``
    """
    if data is None:
        return {records_key: []}
    if isinstance(data, list):
        return {records_key: data}
    if not isinstance(data, dict):
        return data

    primary = data.get(records_key)
    if isinstance(primary, list) and len(primary) > 0:
        return data

    expanded = _expand_wide_column_dict_to_records(data, records_key)
    if expanded is not None:
        return expanded

    alternates = (
        "records",
        "yield_records",
        "data",
        "items",
        "results",
        "rows",
        "extraction_results",
    )
    for alt in alternates:
        if alt == records_key:
            continue
        v = data.get(alt)
        if isinstance(v, list) and len(v) > 0:
            merged = {**data, records_key: v}
            return merged
    return data


_OUTPUT_WRAPPER_BASES: Dict[str, Type[BaseModel]] = {}


def _output_wrapper_base_for(records_key: str) -> Type[BaseModel]:
    """Shared BaseModel subclass with a ``model_validator`` bound to ``records_key``."""
    if records_key in _OUTPUT_WRAPPER_BASES:
        return _OUTPUT_WRAPPER_BASES[records_key]

    rk = records_key

    class _ExtractionOutputWrapperBase(BaseModel):
        @model_validator(mode="before")
        @classmethod
        def _normalize_top_level(cls, data: Any) -> Any:
            return _normalize_output_root_dict(data, rk)

    _ExtractionOutputWrapperBase.__name__ = f"_ExtractionOutputWrapperBase_{rk.replace(' ', '_')}"
    _OUTPUT_WRAPPER_BASES[records_key] = _ExtractionOutputWrapperBase
    return _ExtractionOutputWrapperBase


def _coerce_extract_str(v: Any) -> Optional[str]:
    """
    Normalize LLM JSON values into optional strings.

    Strict JSON parsing often yields numbers for numeric fields; our meta-analysis
    schemas still model every scalar field as ``Optional[str]`` for downstream
    text/ROUGE evaluation. Coerce int/float/bool (and other scalars) to str.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return v
    return str(v)


# Optional[str] that accepts JSON numbers/bools from strict structured output.
ExtractableStr = Annotated[Optional[str], BeforeValidator(_coerce_extract_str)]


class SchemaFactory:
    """
    Factory class for dynamically creating Pydantic models from schema definitions.
    
    Supports:
    - JSON strings (as stored in METADATA_STANDARDS)
    - Python dictionaries
    - Nested structures with field descriptions
    """
    
    def __init__(self):
        self._cache: Dict[str, Type[BaseModel]] = {}
    
    def _parse_schema_string(self, schema_str: str) -> Dict[str, str]:
        """
        Parse a JSON schema string into a dictionary.
        
        Handles:
        - Standard JSON
        - JSON with comments (strips them)
        - Multiline strings
        """
        # Clean the string
        cleaned = schema_str.strip()
        
        # Remove any markdown code blocks if present
        if cleaned.startswith("```json"):
            cleaned = re.sub(r'^```json\s*', '', cleaned)
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```\s*', '', cleaned)
        if cleaned.endswith("```"):
            cleaned = re.sub(r'\s*```$', '', cleaned)
        
        cleaned = cleaned.strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse schema string as JSON: {e}")
    
    def _create_record_model(
        self,
        schema_dict: Dict[str, Any],
        class_name: str = "Record"
    ) -> Type[BaseModel]:
        """
        Create a Pydantic model from a flat dictionary schema.
        
        All fields are treated as Optional[str] since extraction may not
        find all fields for every record.
        
        Args:
            schema_dict: Dictionary mapping field names to descriptions
            class_name: Name for the generated class
            
        Returns:
            A dynamically created Pydantic model class
        """
        # Build field definitions: (type, default) or (type, Field(...))
        field_definitions: Dict[str, Any] = {}
        
        for field_name, description in schema_dict.items():
            # All fields are optional strings with descriptions
            if isinstance(description, str):
                field_definitions[field_name] = (
                    ExtractableStr,
                    Field(default=None, description=description)
                )
            elif isinstance(description, dict):
                # Nested dict - could extend to support nested models
                # For now, store as Optional[Dict]
                field_definitions[field_name] = (
                    Optional[Dict[str, Any]],
                    Field(default=None, description=str(description))
                )
            else:
                # Default to optional string
                field_definitions[field_name] = (
                    ExtractableStr,
                    Field(default=None, description=str(description))
                )
        
        # Create the model dynamically
        model = create_model(class_name, **field_definitions)
        return model
    
    def _create_output_model(
        self,
        record_model: Type[BaseModel],
        class_name: str = "Output",
        records_key: str = "records"
    ) -> Type[BaseModel]:
        """
        Create a wrapper model that contains a list of records.
        
        Args:
            record_model: The Pydantic model for individual records
            class_name: Name for the wrapper class
            records_key: Key name for the records list (e.g., "records", "yield_records")
            
        Returns:
            A dynamically created wrapper Pydantic model
        """
        field_definitions = {
            records_key: (
                List[record_model],
                Field(default_factory=list, description=f"List of {record_model.__name__} entries")
            )
        }

        base = _output_wrapper_base_for(records_key)
        model = create_model(class_name, __base__=base, **field_definitions)
        return model
    
    def create_from_standard(
        self,
        standard: Union[str, Dict[str, Any]],
        record_class_name: str = "Record",
        output_class_name: str = "Output",
        records_key: str = "records"
    ) -> Type[BaseModel]:
        """
        Create a Pydantic output schema from a metadata standard.
        
        Args:
            standard: Either a JSON string or dictionary defining the schema fields
            record_class_name: Name for the individual record model class
            output_class_name: Name for the wrapper output model class
            records_key: Key name for the list of records in the output
            
        Returns:
            A Pydantic model class that can be used as output_schema
            
        Example:
            >>> factory = SchemaFactory()
            >>> OutputSchema = factory.create_from_standard(
            ...     METADATA_STANDARDS["climate_vs_cropyield"],
            ...     record_class_name="YieldRecord",
            ...     output_class_name="MetaAnalysisOutput",
            ...     records_key="records"
            ... )
            >>> # OutputSchema is now a Pydantic model with structure:
            >>> # class MetaAnalysisOutput(BaseModel):
            >>> #     records: List[YieldRecord]
        """
        # Parse if string
        if isinstance(standard, str):
            schema_dict = self._parse_schema_string(standard)
        else:
            schema_dict = standard
        
        # Create cache key
        cache_key = f"{output_class_name}_{records_key}_{hash(frozenset(schema_dict.keys()))}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Create record model
        record_model = self._create_record_model(schema_dict, record_class_name)
        
        # Create output wrapper model
        output_model = self._create_output_model(
            record_model,
            output_class_name,
            records_key
        )
        
        # Cache and return
        self._cache[cache_key] = output_model
        return output_model
    
    def create_record_only(
        self,
        standard: Union[str, Dict[str, Any]],
        class_name: str = "Record"
    ) -> Type[BaseModel]:
        """
        Create only the record model (without the wrapper).
        
        Useful if you need to work with individual records.
        
        Args:
            standard: Either a JSON string or dictionary defining the schema fields
            class_name: Name for the record model class
            
        Returns:
            A Pydantic model class for individual records
        """
        if isinstance(standard, str):
            schema_dict = self._parse_schema_string(standard)
        else:
            schema_dict = standard
        
        return self._create_record_model(schema_dict, class_name)
    
    def get_field_names(self, standard: Union[str, Dict[str, Any]]) -> List[str]:
        """
        Extract field names from a standard definition.
        
        Args:
            standard: Either a JSON string or dictionary
            
        Returns:
            List of field names defined in the schema
        """
        if isinstance(standard, str):
            schema_dict = self._parse_schema_string(standard)
        else:
            schema_dict = standard
        
        return list(schema_dict.keys())


# Convenience function for quick schema creation
def create_output_schema(
    standard: Union[str, Dict[str, Any]],
    record_class_name: str = "Record",
    output_class_name: str = "Output",
    records_key: str = "records"
) -> Type[BaseModel]:
    """
    Convenience function to create an output schema from a standard.
    
    Args:
        standard: Either a JSON string or dictionary defining the schema fields
        record_class_name: Name for the individual record model class
        output_class_name: Name for the wrapper output model class
        records_key: Key name for the list of records in the output
        
    Returns:
        A Pydantic model class that can be used as output_schema
        
    Example:
        >>> from src.standards import METADATA_STANDARDS
        >>> from src.core.schema_factory import create_output_schema
        >>> 
        >>> OutputSchema = create_output_schema(
        ...     METADATA_STANDARDS["climate_vs_cropyield"],
        ...     record_class_name="YieldRecord",
        ...     output_class_name="MetaAnalysisOutput"
        ... )
    """
    factory = SchemaFactory()
    return factory.create_from_standard(
        standard,
        record_class_name=record_class_name,
        output_class_name=output_class_name,
        records_key=records_key
    )
