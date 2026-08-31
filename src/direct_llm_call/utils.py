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
    PROVIDER_CONFIGS,
    create_llm,
    get_model_name,
    structured_output_kwargs,
    iter_structured_output_kwargs,
)

from .schemas import create_extraction_schema


T = TypeVar("T", bound=BaseModel)


def _ensure_json_keyword_for_structured_prompt(prompt: str) -> str:
    """
    Some OpenAI-compatible backends reject `response_format=json_object`
    unless the prompt/messages explicitly contain the word "json".
    """
    text = prompt or ""
    if "json" in text.lower():
        return text
    return (
        "IMPORTANT: Return a valid JSON object that matches the schema.\n\n"
        + text
    )


def _ensure_top_level_records_wrapper(prompt: str, records_key: str) -> str:
    """
    Reinforce that output must be a JSON object with the expected top-level key.

    This is especially important for `json_mode` providers (e.g., some
    OpenAI-compatible Qwen endpoints), which may otherwise return a bare array.
    """
    text = prompt or ""
    marker = f"\"{records_key}\""
    if marker in text or f"`{records_key}`" in text:
        return text
    return (
        f"IMPORTANT: Return a top-level JSON object with key "
        f"\"{records_key}\" only, e.g. {{\"{records_key}\": [ ... ]}}. "
        f"Do NOT return a bare array.\n\n"
        + text
    )


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
    llm = create_llm(
        model_name=model_name,
        temperature=temperature,
        provider=provider,
    )
    return llm.with_structured_output(
        output_schema, **structured_output_kwargs(provider)
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
    records_key: Optional[str] = None,
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
        records_key: When set, reinforce top-level JSON object shape (see
            ``_ensure_top_level_records_wrapper``). Use the wrapper model's list
            field name (e.g. ``yield_records``, ``facts``).

    Returns:
        An instance of the output_schema populated with the LLM's response
    """
    llm = create_llm(
        model_name=model_name,
        temperature=temperature,
        provider=provider,
    )
    prompt = _ensure_json_keyword_for_structured_prompt(prompt)
    if records_key:
        prompt = _ensure_top_level_records_wrapper(prompt, records_key)

    last_error: Optional[Exception] = None
    for kwargs in iter_structured_output_kwargs(provider):
        try:
            bound = llm.with_structured_output(output_schema, **kwargs)
            return bound.invoke(prompt)
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("Structured output invocation failed with no error")


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
    output_schema = create_extraction_schema(
        standard=schema,
        record_class_name=record_class_name,
        output_class_name=output_class_name,
        records_key=records_key,
    )
    return invoke_llm_with_structured_output(
        prompt=prompt,
        output_schema=output_schema,
        model_name=model_name,
        temperature=temperature,
        provider=provider,
        records_key=records_key,
    )


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
