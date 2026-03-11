"""
Standard filtering utilities for meta-analysis experiments.

This module provides functions to filter metadata standards by removing
unnecessary fields when running experiments with reduced schemas.
"""
import json
import re
from typing import Any, Dict, List, Optional, Union


def _parse_standard(standard: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parse a standard (JSON string or dict) into a dictionary.
    
    Handles JSON strings as stored in METADATA_STANDARDS.
    """
    if isinstance(standard, dict):
        return dict(standard)
    
    cleaned = standard.strip()
    if cleaned.startswith("```json"):
        cleaned = re.sub(r'^```json\s*', '', cleaned)
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```\s*', '', cleaned)
    if cleaned.endswith("```"):
        cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()
    
    return json.loads(cleaned)


def filter_standard(
    standard: Union[str, Dict[str, Any]],
    exclude_keys: List[str],
    *,
    include_keys: Optional[List[str]] = None,
    as_dict: bool = False,
) -> Union[str, Dict[str, Any]]:
    """
    Filter a metadata standard by removing unnecessary key-value pairs.
    
    Use this to create reduced schemas for experiments (e.g., removing
    *_source_section, *_confidence, *_notes fields to simplify extraction).
    
    Args:
        standard: Metadata standard as JSON string or dict (e.g., from
            METADATA_STANDARDS["climate_vs_cropyield"]).
        exclude_keys: Keys to remove from the standard.
        include_keys: If provided, only these keys are kept (exclude_keys
            is ignored). Use for whitelist-style filtering.
        as_dict: If True, return a dict. If False, return the same type
            as input (string for string input, dict for dict input).
            
    Returns:
        Filtered standard in the same format as input (unless as_dict=True).
        
    Example:
        >>> from src.standards import METADATA_STANDARDS
        >>> from src.experimentutils.standard_utils import filter_standard
        >>> 
        >>> # Remove metadata fields (source_section, confidence, notes)
        >>> reduced = filter_standard(
        ...     METADATA_STANDARDS["climate_vs_cropyield"],
        ...     exclude_keys=[
        ...         "yield_source_section", "yield_confidence", "yield_notes",
        ...         "Treatment_source_section", "Treatment_confidence", "Treatment_notes",
        ...         "Tillage_source_section", "Tillage_confidence", "Tillage_notes",
        ...         "Soil_source_section", "Soil_confidence", "Soil_notes",
        ...         "climate_source_section", "climate_confidence", "climate_notes",
        ...         "rs_source_section", "rs_confidence", "rs_notes",
        ...     ],
        ... )
        >>> 
        >>> # Or keep only core fields (whitelist)
        >>> core_only = filter_standard(
        ...     METADATA_STANDARDS["climate_vs_cropyield"],
        ...     exclude_keys=[],  # ignored when include_keys is set
        ...     include_keys=["crop_type", "yield_value", "location", "year"],
        ... )
    """
    parsed = _parse_standard(standard)
    
    if include_keys is not None:
        filtered = {k: v for k, v in parsed.items() if k in include_keys}
    else:
        exclude_set = set(exclude_keys)
        filtered = {k: v for k, v in parsed.items() if k not in exclude_set}
    
    if as_dict:
        return filtered
    
    # Match input type
    if isinstance(standard, dict):
        return filtered
    
    return json.dumps(filtered, indent=4)


def get_metadata_field_keys(standard: Union[str, Dict[str, Any]]) -> List[str]:
    """
    Return keys that match common metadata patterns (*_source_section,
    *_confidence, *_notes) for use with filter_standard(exclude_keys=...).
    
    Useful when you want to strip all metadata fields in one call.
    
    Example:
        >>> keys = get_metadata_field_keys(METADATA_STANDARDS["climate_vs_cropyield"])
        >>> reduced = filter_standard(standard, exclude_keys=keys)
    """
    parsed = _parse_standard(standard)
    metadata_suffixes = ("_source_section", "_confidence", "_notes")
    return [
        k for k in parsed
        if any(k.endswith(suffix) for suffix in metadata_suffixes)
    ]


__all__ = [
    "filter_standard",
    "get_metadata_field_keys",
]
