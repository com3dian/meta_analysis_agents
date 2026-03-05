"""
Utility functions for direct LLM calls with structured output.

This module provides functions for:
- Creating LLM instances with structured output support
- Reading markdown files
- Invoking LLMs with Pydantic schema-based structured output
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

from pydantic import BaseModel

# Import config from parent module
from src.config import (
    LLM_PROVIDER,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    SURF_API_BASE,
    SURF_API_KEY,
    QWEN_API_BASE,
    QWEN_API_KEY,
    PROVIDER_CONFIGS,
    get_model_name,
)

from .schemas import create_extraction_schema


T = TypeVar("T", bound=BaseModel)


def read_markdown_file(file_path: str) -> str:
    """
    Read the contents of a markdown file.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        The content of the file as a string
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is not a markdown file
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if path.suffix.lower() not in [".md", ".markdown"]:
        raise ValueError(f"Expected a markdown file (.md), got: {path.suffix}")
    
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def create_llm_with_structured_output(
    output_schema: Type[T],
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    provider: Optional[str] = None,
):
    """
    Create an LLM instance configured for structured output.
    
    Args:
        output_schema: A Pydantic model class defining the expected output structure
        model_name: Optional model name override
        temperature: LLM temperature (default: 0.0 for deterministic output)
        provider: Optional provider override (google, openai, surf, qwen)
        
    Returns:
        An LLM instance with structured output binding
        
    Raises:
        ValueError: If provider is not supported or required config is missing
    """
    provider = provider or LLM_PROVIDER
    model = get_model_name(model_name)
    
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not found. Set it in your .env file."
            )
        
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=GOOGLE_API_KEY,
        )
        return llm.with_structured_output(output_schema)
    
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it in your .env file."
            )
        
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY,
        )
        return llm.with_structured_output(output_schema)
    
    elif provider == "surf":
        from langchain_openai import ChatOpenAI
        
        if not SURF_API_BASE:
            raise ValueError(
                "SURF_API_BASE not found. Set it in your .env file.\n"
                "Example: SURF_API_BASE=http://localhost:8000/v1"
            )
        
        if not SURF_API_KEY:
            raise ValueError(
                "SURF_API_KEY not found. Set it in your .env file."
            )
        
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=SURF_API_KEY,
            openai_api_base=SURF_API_BASE,
        )
        return llm.with_structured_output(output_schema)
    
    elif provider == "qwen":
        from langchain_openai import ChatOpenAI
        
        if not QWEN_API_BASE:
            raise ValueError(
                "QWEN_API_BASE not found. Set it in your .env file.\n"
                "Example: QWEN_API_BASE=http://localhost:8000/v1"
            )
        
        if not QWEN_API_KEY:
            raise ValueError(
                "QWEN_API_KEY not found. Set it in your .env file."
            )
        
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=QWEN_API_KEY,
            openai_api_base=QWEN_API_BASE,
        )
        return llm.with_structured_output(output_schema)
    
    else:
        available = list(PROVIDER_CONFIGS.keys())
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. Available: {available}"
        )


def create_llm_for_schema(
    schema: Union[str, Dict[str, Any]],
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    provider: Optional[str] = None,
    record_class_name: str = "Record",
    output_class_name: str = "ExtractionResult",
    records_key: str = "yield_records",
):
    """
    Create an LLM with structured output from a schema definition.
    
    This is a convenience function that combines schema creation and LLM setup.
    It accepts the same schema format as METADATA_STANDARDS.
    
    Args:
        schema: Either a JSON string or dictionary defining the schema fields.
                Can use METADATA_STANDARDS["climate_vs_cropyield"] directly.
        model_name: Optional model name override
        temperature: LLM temperature (default: 0.0)
        provider: Optional provider override (google, openai, surf, qwen)
        record_class_name: Name for the individual record model class
        output_class_name: Name for the wrapper output model class
        records_key: Key name for the list of records in the output
        
    Returns:
        Tuple of (llm_with_structured_output, pydantic_schema_class)
        
    Example:
        >>> from src.standards import METADATA_STANDARDS
        >>> 
        >>> llm, Schema = create_llm_for_schema(
        ...     METADATA_STANDARDS["climate_vs_cropyield"],
        ...     provider="google"
        ... )
        >>> result = llm.invoke(prompt)
    """
    # Create the Pydantic schema
    output_schema = create_extraction_schema(
        standard=schema,
        record_class_name=record_class_name,
        output_class_name=output_class_name,
        records_key=records_key
    )
    
    # Create LLM with structured output
    llm = create_llm_with_structured_output(
        output_schema=output_schema,
        model_name=model_name,
        temperature=temperature,
        provider=provider,
    )
    
    return llm, output_schema


def invoke_llm_with_structured_output(
    prompt: str,
    output_schema: Type[T],
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    provider: Optional[str] = None,
) -> T:
    """
    Invoke an LLM with a prompt and return structured output.
    
    This is a convenience function that combines LLM creation and invocation.
    
    Args:
        prompt: The prompt string to send to the LLM
        output_schema: A Pydantic model class defining the expected output structure
        model_name: Optional model name override
        temperature: LLM temperature (default: 0.0 for deterministic output)
        provider: Optional provider override (google, openai, surf, qwen)
        
    Returns:
        An instance of the output_schema populated with the LLM's response
    """
    llm = create_llm_with_structured_output(
        output_schema=output_schema,
        model_name=model_name,
        temperature=temperature,
        provider=provider,
    )
    
    return llm.invoke(prompt)


def invoke_with_schema(
    prompt: str,
    schema: Union[str, Dict[str, Any]],
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    provider: Optional[str] = None,
    record_class_name: str = "Record",
    output_class_name: str = "ExtractionResult",
    records_key: str = "yield_records",
) -> BaseModel:
    """
    Invoke an LLM with a prompt using a schema definition.
    
    This is a convenience function that handles schema creation and LLM invocation
    in one call. Accepts the same schema format as METADATA_STANDARDS.
    
    Args:
        prompt: The prompt string to send to the LLM
        schema: Either a JSON string or dictionary defining the schema fields
        model_name: Optional model name override
        temperature: LLM temperature (default: 0.0)
        provider: Optional provider override
        record_class_name: Name for the individual record model class
        output_class_name: Name for the wrapper output model class
        records_key: Key name for the list of records in the output
        
    Returns:
        A Pydantic model instance populated with the LLM's response
    """
    llm, _ = create_llm_for_schema(
        schema=schema,
        model_name=model_name,
        temperature=temperature,
        provider=provider,
        record_class_name=record_class_name,
        output_class_name=output_class_name,
        records_key=records_key,
    )
    
    return llm.invoke(prompt)


def get_provider_info() -> Dict[str, Any]:
    """
    Get information about the current LLM provider configuration.
    
    Returns:
        A dictionary with provider configuration details
    """
    provider_config = PROVIDER_CONFIGS.get(LLM_PROVIDER, {})
    model = get_model_name()
    
    # Check API key status
    api_key_env = provider_config.get("api_key_env", "")
    api_key_set = bool(os.getenv(api_key_env)) if api_key_env else False
    
    return {
        "provider": LLM_PROVIDER,
        "description": provider_config.get("description", "Unknown"),
        "model": model,
        "api_key_configured": api_key_set,
    }


# Export utilities
__all__ = [
    "read_markdown_file",
    "create_llm_with_structured_output",
    "create_llm_for_schema",
    "invoke_llm_with_structured_output",
    "invoke_with_schema",
    "get_provider_info",
]
