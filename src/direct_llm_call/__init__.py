"""
Direct LLM Call for Meta-Analysis Extraction.

This module provides a simple, single-call LLM approach for extracting
meta-analysis data from research papers. It serves as a baseline comparison
to the multi-agent system approach.

The module supports flexible schema definitions using the same format as
METADATA_STANDARDS, allowing dynamic Pydantic model generation.

Usage:
    from src.direct_llm_call import extract_meta_analysis
    from src.standards import METADATA_STANDARDS
    
    # Using a predefined standard
    result = extract_meta_analysis(
        "path/to/paper.md",
        schema=METADATA_STANDARDS["climate_vs_cropyield"]
    )
    print(result.yield_records)
    
    # Using a custom schema dictionary
    custom_schema = {
        "crop_type": "name of the crop",
        "yield_value": "measured yield with units",
        "location": "study location"
    }
    result = extract_meta_analysis(
        "path/to/paper.md",
        schema=custom_schema
    )
    
    # With custom options:
    result = extract_meta_analysis(
        "path/to/paper.md",
        schema=METADATA_STANDARDS["climate_vs_cropyield"],
        model_name="gpt-4o",
        provider="openai"
    )
"""
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

# Import components
from .schemas import (
    create_extraction_schema,
    create_record_schema,
    get_schema_field_names,
    format_schema_for_prompt,
    get_schema_description,
    SchemaFactory,
)
from .prompts import (
    get_extraction_prompt,
    get_simple_extraction_prompt,
    get_custom_extraction_prompt,
    META_ANALYSIS_EXTRACTION_PROMPT_TEMPLATE,
)
from .utils import (
    read_markdown_file,
    create_llm_with_structured_output,
    create_llm_for_schema,
    invoke_llm_with_structured_output,
    invoke_with_schema,
    get_provider_info,
)

# Import standards for convenience
from src.standards import METADATA_STANDARDS


# Default schema to use if none specified
DEFAULT_SCHEMA = METADATA_STANDARDS["climate_vs_cropyield"]


def extract_meta_analysis(
    md_file_path: str,
    schema: Union[str, Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    provider: Optional[str] = None,
    records_key: str = "yield_records",
    prompt_style: str = "full",
) -> BaseModel:
    """
    Extract meta-analysis data from a markdown file using a single LLM call.
    
    This function reads a research paper in markdown format, sends it to an LLM
    with a structured extraction prompt, and returns the extracted records
    as a dynamically-generated Pydantic model.
    
    Args:
        md_file_path: Path to the markdown file containing the research paper
        schema: Schema definition (JSON string or dict). Uses METADATA_STANDARDS format.
                If None, uses the default "climate_vs_cropyield" schema.
        model_name: Optional LLM model name override (uses config default if not specified)
        temperature: LLM temperature, default 0.0 for deterministic extraction
        provider: Optional LLM provider override (google, openai, surf, qwen)
        records_key: Key name for the records list in output (default: "yield_records")
        prompt_style: Style of prompt to use: "full", "simple", or "custom"
        
    Returns:
        A Pydantic model instance containing:
            - {records_key}: List of extracted record objects
            
    Raises:
        FileNotFoundError: If the markdown file doesn't exist
        ValueError: If the file is not a markdown file or LLM config is invalid
        
    Example:
        >>> from src.standards import METADATA_STANDARDS
        >>> 
        >>> # Using predefined schema
        >>> result = extract_meta_analysis(
        ...     "data/papers/mason_1986.md",
        ...     schema=METADATA_STANDARDS["climate_vs_cropyield"]
        ... )
        >>> for record in result.yield_records:
        ...     print(f"{record.crop_type}: {record.yield_value}")
        
        >>> # Using custom schema
        >>> result = extract_meta_analysis(
        ...     "data/papers/paper.md",
        ...     schema={"crop": "crop name", "yield": "yield value"}
        ... )
    """
    # Use default schema if none provided
    if schema is None:
        schema = DEFAULT_SCHEMA
    
    # Read the markdown file
    document_content = read_markdown_file(md_file_path)
    
    # Format the extraction prompt with schema
    if prompt_style == "simple":
        prompt = get_simple_extraction_prompt(document_content, schema)
    elif prompt_style == "custom":
        prompt = get_custom_extraction_prompt(document_content, schema)
    else:
        prompt = get_extraction_prompt(document_content, schema)
    
    # Invoke the LLM with structured output
    result = invoke_with_schema(
        prompt=prompt,
        schema=schema,
        model_name=model_name,
        temperature=temperature,
        provider=provider,
        record_class_name="Record",
        output_class_name="ExtractionResult",
        records_key=records_key,
    )
    
    return result


def extract_meta_analysis_to_dict(
    md_file_path: str,
    schema: Union[str, Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    provider: Optional[str] = None,
    records_key: str = "yield_records",
) -> Dict[str, Any]:
    """
    Extract meta-analysis data and return as a dictionary.
    
    This is a convenience wrapper around extract_meta_analysis() that returns
    the result as a dictionary instead of a Pydantic model. Useful for JSON
    serialization or compatibility with existing code.
    
    Args:
        md_file_path: Path to the markdown file containing the research paper
        schema: Schema definition (JSON string or dict). Uses METADATA_STANDARDS format.
        model_name: Optional LLM model name override
        temperature: LLM temperature, default 0.0
        provider: Optional LLM provider override
        records_key: Key name for the records list
        
    Returns:
        Dictionary containing the extraction result
    """
    result = extract_meta_analysis(
        md_file_path=md_file_path,
        schema=schema,
        model_name=model_name,
        temperature=temperature,
        provider=provider,
        records_key=records_key,
    )
    return result.model_dump()


def extract_records_only(
    md_file_path: str,
    schema: Union[str, Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    provider: Optional[str] = None,
    records_key: str = "yield_records",
) -> List[Dict[str, Any]]:
    """
    Extract only the records from a paper as a list of dictionaries.
    
    This is useful when you only need the records and want them in
    a format compatible with pandas or CSV export.
    
    Args:
        md_file_path: Path to the markdown file containing the research paper
        schema: Schema definition (JSON string or dict). Uses METADATA_STANDARDS format.
        model_name: Optional LLM model name override
        temperature: LLM temperature, default 0.0
        provider: Optional LLM provider override
        records_key: Key name for the records list
        
    Returns:
        List of record dictionaries
    """
    result = extract_meta_analysis(
        md_file_path=md_file_path,
        schema=schema,
        model_name=model_name,
        temperature=temperature,
        provider=provider,
        records_key=records_key,
    )
    # Get the records from the result using the records_key
    records = getattr(result, records_key, [])
    return [record.model_dump() for record in records]


def get_schema_for_standard(
    standard_name: str,
    record_class_name: str = "Record",
    output_class_name: str = "ExtractionResult",
    records_key: str = "yield_records",
) -> Type[BaseModel]:
    """
    Get a Pydantic schema class for a predefined metadata standard.
    
    Args:
        standard_name: Name of the standard (e.g., "climate_vs_cropyield")
        record_class_name: Name for the individual record model class
        output_class_name: Name for the wrapper output model class
        records_key: Key name for the list of records in the output
        
    Returns:
        A dynamically created Pydantic model class
        
    Raises:
        KeyError: If the standard_name is not found in METADATA_STANDARDS
    """
    if standard_name not in METADATA_STANDARDS:
        available = list(METADATA_STANDARDS.keys())
        raise KeyError(
            f"Unknown standard: '{standard_name}'. Available: {available}"
        )
    
    return create_extraction_schema(
        standard=METADATA_STANDARDS[standard_name],
        record_class_name=record_class_name,
        output_class_name=output_class_name,
        records_key=records_key,
    )


# Export public API
__all__ = [
    # Main extraction functions
    "extract_meta_analysis",
    "extract_meta_analysis_to_dict",
    "extract_records_only",
    
    # Schema utilities
    "create_extraction_schema",
    "create_record_schema",
    "get_schema_field_names",
    "format_schema_for_prompt",
    "get_schema_description",
    "get_schema_for_standard",
    "SchemaFactory",
    
    # Prompt utilities
    "get_extraction_prompt",
    "get_simple_extraction_prompt",
    "get_custom_extraction_prompt",
    "META_ANALYSIS_EXTRACTION_PROMPT_TEMPLATE",
    
    # LLM utilities
    "create_llm_with_structured_output",
    "create_llm_for_schema",
    "invoke_llm_with_structured_output",
    "invoke_with_schema",
    "get_provider_info",
    "read_markdown_file",
    
    # Standards (re-exported for convenience)
    "METADATA_STANDARDS",
    "DEFAULT_SCHEMA",
]
