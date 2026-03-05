"""
Schema utilities for direct LLM extraction.

This module provides schema creation and formatting utilities that work with
the METADATA_STANDARDS dictionary format, creating dynamic Pydantic models.

Uses the SchemaFactory from src/core/schema_factory.py for consistency
with the multi-agent approach.
"""
from typing import Any, Dict, List, Type, Union

from pydantic import BaseModel

# Import the shared SchemaFactory
from src.core.schema_factory import SchemaFactory, create_output_schema


# Singleton factory instance for caching
_factory = SchemaFactory()


def create_extraction_schema(
    standard: Union[str, Dict[str, Any]],
    record_class_name: str = "Record",
    output_class_name: str = "ExtractionResult",
    records_key: str = "yield_records"
) -> Type[BaseModel]:
    """
    Create a Pydantic schema from a metadata standard definition.
    
    This is the main entry point for creating extraction schemas in the
    direct LLM call approach. It uses the same SchemaFactory as the
    multi-agent system for consistency.
    
    Args:
        standard: Either a JSON string or dictionary defining the schema fields.
                  Can use METADATA_STANDARDS["climate_vs_cropyield"] directly.
        record_class_name: Name for the individual record model class
        output_class_name: Name for the wrapper output model class
        records_key: Key name for the list of records in the output
        
    Returns:
        A dynamically created Pydantic model class
        
    Example:
        >>> from src.standards import METADATA_STANDARDS
        >>> 
        >>> Schema = create_extraction_schema(
        ...     METADATA_STANDARDS["climate_vs_cropyield"],
        ...     record_class_name="YieldRecord",
        ...     output_class_name="MetaAnalysisResult"
        ... )
    """
    return _factory.create_from_standard(
        standard=standard,
        record_class_name=record_class_name,
        output_class_name=output_class_name,
        records_key=records_key
    )


def create_record_schema(
    standard: Union[str, Dict[str, Any]],
    class_name: str = "Record"
) -> Type[BaseModel]:
    """
    Create only the record model (without the list wrapper).
    
    Useful when you need to work with individual records.
    
    Args:
        standard: Either a JSON string or dictionary defining the schema fields
        class_name: Name for the record model class
        
    Returns:
        A Pydantic model class for individual records
    """
    return _factory.create_record_only(standard, class_name)


def get_schema_field_names(standard: Union[str, Dict[str, Any]]) -> List[str]:
    """
    Extract field names from a standard definition.
    
    Args:
        standard: Either a JSON string or dictionary
        
    Returns:
        List of field names defined in the schema
    """
    return _factory.get_field_names(standard)


def format_schema_for_prompt(standard: Union[str, Dict[str, Any]]) -> str:
    """
    Format a schema definition for inclusion in a prompt.
    
    This converts the schema to a clean, readable format that can be
    embedded in the extraction prompt to guide the LLM.
    
    Args:
        standard: Either a JSON string or dictionary defining the schema fields
        
    Returns:
        A formatted string representation of the schema suitable for prompts
        
    Example:
        >>> schema_str = format_schema_for_prompt(METADATA_STANDARDS["climate_vs_cropyield"])
        >>> print(schema_str)
        {
            "crop_type": "specific crop name (e.g., maize, wheat, rice...)",
            ...
        }
    """
    import json
    
    if isinstance(standard, str):
        # Already a string, just clean it up
        cleaned = standard.strip()
        # Try to parse and re-format for consistent indentation
        try:
            parsed = json.loads(cleaned)
            return json.dumps(parsed, indent=4, ensure_ascii=False)
        except json.JSONDecodeError:
            # Return as-is if not valid JSON
            return cleaned
    elif isinstance(standard, dict):
        return json.dumps(standard, indent=4, ensure_ascii=False)
    else:
        return str(standard)


def get_schema_description(standard: Union[str, Dict[str, Any]]) -> str:
    """
    Generate a human-readable description of schema fields.
    
    This creates a bulleted list of fields and their descriptions,
    useful for documentation or verbose prompts.
    
    Args:
        standard: Either a JSON string or dictionary defining the schema fields
        
    Returns:
        A formatted string with field descriptions
    """
    import json
    
    if isinstance(standard, str):
        try:
            schema_dict = json.loads(standard.strip())
        except json.JSONDecodeError:
            return "Unable to parse schema"
    else:
        schema_dict = standard
    
    lines = ["Schema Fields:"]
    for field_name, description in schema_dict.items():
        lines.append(f"  - **{field_name}**: {description}")
    
    return "\n".join(lines)


# Re-export for convenience
__all__ = [
    "create_extraction_schema",
    "create_record_schema",
    "get_schema_field_names",
    "format_schema_for_prompt",
    "get_schema_description",
    "SchemaFactory",
    "create_output_schema",
]
