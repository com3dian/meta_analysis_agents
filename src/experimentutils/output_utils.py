"""
Output utilities for meta-analysis experiments.

This module provides functions for:
- Saving experiment results to organized directories
- Date-prefixed output paths for tracking experiments
- CSV export with consistent formatting
- Parsing and saving JSON records from LLM / agent outputs
"""
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


# Default output directory (project-level `outputs` folder)
# This is resolved relative to the project root (one level above `src`),
# so that notebooks or scripts can be run from any working directory
# and still write to a shared `outputs` directory at the repository root.
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"


def get_dated_output_dir(
    base_dir: str = DEFAULT_OUTPUT_DIR,
    date: Optional[str] = None,
    create: bool = True,
) -> str:
    """
    Get the output directory path with date prefix.
    
    Args:
        base_dir: Base output directory (default: "outputs")
        date: Date string in YYYY-MM-DD format. If None, uses today's date.
        create: If True, create the directory if it doesn't exist.
        
    Returns:
        Path to the dated output directory (e.g., "outputs/2026-03-09/")
        
    Example:
        >>> path = get_dated_output_dir()
        >>> print(path)
        'outputs/2026-03-09'
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    output_path = Path(base_dir) / date
    
    if create:
        output_path.mkdir(parents=True, exist_ok=True)
    
    return str(output_path)


def get_output_path(
    filename: str,
    base_dir: str = DEFAULT_OUTPUT_DIR,
    date: Optional[str] = None,
    create_dir: bool = True,
) -> str:
    """
    Get the full output path for a file with date prefix.
    
    Args:
        filename: Name of the output file (e.g., "results.csv")
        base_dir: Base output directory (default: "outputs")
        date: Date string in YYYY-MM-DD format. If None, uses today's date.
        create_dir: If True, create the directory if it doesn't exist.
        
    Returns:
        Full path to the output file (e.g., "outputs/2026-03-09/results.csv")
        
    Example:
        >>> path = get_output_path("extraction_results.csv")
        >>> print(path)
        'outputs/2026-03-09/extraction_results.csv'
    """
    dated_dir = get_dated_output_dir(base_dir, date, create=create_dir)
    return str(Path(dated_dir) / filename)


def save_records_to_csv(
    records: Union[List[Dict[str, Any]], pd.DataFrame],
    filename: str,
    base_dir: str = DEFAULT_OUTPUT_DIR,
    date: Optional[str] = None,
    append: bool = False,
) -> str:
    """
    Save records to a CSV file in the dated output directory.
    
    Args:
        records: List of record dictionaries or a pandas DataFrame
        filename: Name of the output file (e.g., "results.csv")
        base_dir: Base output directory (default: "outputs")
        date: Date string in YYYY-MM-DD format. If None, uses today's date.
        append: If True, append to existing file. If False, overwrite.
        
    Returns:
        Path to the saved CSV file
        
    Example:
        >>> records = [{"crop": "maize", "yield": "8.5 t/ha"}]
        >>> path = save_records_to_csv(records, "results.csv")
        >>> print(path)
        'outputs/2026-03-09/results.csv'
    """
    output_path = get_output_path(filename, base_dir, date)
    
    # Convert to DataFrame if needed
    if isinstance(records, list):
        df = pd.DataFrame(records)
    else:
        df = records
    
    # Handle append mode
    if append and os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(output_path, index=False)
    
    return output_path


def save_extraction_results(
    results: Any,
    filename: str,
    base_dir: str = DEFAULT_OUTPUT_DIR,
    date: Optional[str] = None,
    records_key: str = "yield_records",
) -> str:
    """
    Save extraction results (from direct_llm_call or orchestrator) to CSV.
    
    This function handles both Pydantic models and dictionaries.
    
    Args:
        results: Extraction results (Pydantic model or dict with records)
        filename: Name of the output file
        base_dir: Base output directory (default: "outputs")
        date: Date string. If None, uses today's date.
        records_key: Key to extract records from results dict
        
    Returns:
        Path to the saved CSV file
        
    Example:
        >>> from src.direct_llm_call import extract_meta_analysis
        >>> result = extract_meta_analysis("paper.md")
        >>> path = save_extraction_results(result, "paper_results.csv")
    """
    # Handle Pydantic model
    if hasattr(results, 'model_dump'):
        results_dict = results.model_dump()
    elif hasattr(results, 'dict'):
        results_dict = results.dict()
    elif isinstance(results, dict):
        results_dict = results
    else:
        raise ValueError(f"Unsupported results type: {type(results)}")
    
    # Extract records
    if records_key in results_dict:
        records = results_dict[records_key]
    elif 'records' in results_dict:
        records = results_dict['records']
    elif isinstance(results_dict, list):
        records = results_dict
    else:
        # Try to find any list in the dict
        for key, value in results_dict.items():
            if isinstance(value, list):
                records = value
                break
        else:
            records = [results_dict]
    
    return save_records_to_csv(records, filename, base_dir, date)


def save_extraction_results_with_timestamp(
    results: Any,
    base_name: str = "extraction_results",
    base_dir: str = DEFAULT_OUTPUT_DIR,
    date: Optional[str] = None,
    records_key: str = "yield_records",
    include_time: bool = False,
) -> str:
    """
    Convenience wrapper to save extraction results to a timestamped CSV.
    
    This combines timestamped filename generation and saving into a single step,
    so callers do not need to manually call both get_timestamped_filename and
    save_extraction_results.
    
    Args:
        results: Extraction results object (Pydantic model, dict, or list)
        base_name: Base name for the output file (without extension)
        base_dir: Base output directory (defaults to project-level "outputs")
        date: Date string for subdirectory. If None, uses today's date.
        records_key: Key under which records are stored in results
        include_time: If True, include time in the filename
        
    Returns:
        Full path to the saved CSV file.
    """
    filename = get_timestamped_filename(
        base_name=base_name,
        extension="csv",
        include_time=include_time,
    )
    return save_extraction_results(
        results=results,
        filename=filename,
        base_dir=str(base_dir),
        date=date,
        records_key=records_key,
    )


def get_timestamped_filename(
    base_name: str,
    extension: str = "csv",
    include_time: bool = False,
) -> str:
    """
    Generate a timestamped filename.
    
    Args:
        base_name: Base name for the file (e.g., "extraction_results")
        extension: File extension without dot (default: "csv")
        include_time: If True, include time in filename (HH-MM-SS)
        
    Returns:
        Timestamped filename (e.g., "extraction_results_2026-03-09.csv")
        
    Example:
        >>> name = get_timestamped_filename("results")
        >>> print(name)
        'results_2026-03-09.csv'
        
        >>> name = get_timestamped_filename("results", include_time=True)
        >>> print(name)
        'results_2026-03-09_14-30-45.csv'
    """
    if include_time:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d")
    
    return f"{base_name}_{timestamp}.{extension}"


def list_output_files(
    base_dir: str = DEFAULT_OUTPUT_DIR,
    date: Optional[str] = None,
    pattern: str = "*.csv",
) -> List[str]:
    """
    List output files in the dated directory.
    
    Args:
        base_dir: Base output directory
        date: Date string. If None, uses today's date.
        pattern: Glob pattern for files (default: "*.csv")
        
    Returns:
        List of file paths matching the pattern
    """
    dated_dir = get_dated_output_dir(base_dir, date, create=False)
    
    if not os.path.exists(dated_dir):
        return []
    
    return sorted(str(p) for p in Path(dated_dir).glob(pattern))


def clean_old_outputs(
    base_dir: str = DEFAULT_OUTPUT_DIR,
    keep_days: int = 7,
) -> List[str]:
    """
    Remove output directories older than keep_days.
    
    Args:
        base_dir: Base output directory
        keep_days: Number of days to keep (default: 7)
        
    Returns:
        List of removed directory paths
    """
    import shutil
    
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    removed = []
    cutoff_date = datetime.now().date()
    
    for item in base_path.iterdir():
        if item.is_dir():
            try:
                # Try to parse directory name as date
                dir_date = datetime.strptime(item.name, "%Y-%m-%d").date()
                days_old = (cutoff_date - dir_date).days
                
                if days_old > keep_days:
                    shutil.rmtree(item)
                    removed.append(str(item))
            except ValueError:
                # Not a date-formatted directory, skip
                continue
    
    return removed


def get_method_output_path(
    paper_index: int,
    method_name: str,
    n_fields: int,
    base_dir: str = DEFAULT_OUTPUT_DIR,
    date: Optional[str] = None,
) -> str:
    """
    Build a dated output path for an extraction method run on a specific paper.

    Filename pattern: ``{paper_index}_{method_name}_{n_fields}fields_{date}.csv``

    Args:
        paper_index: 1-based index of the paper (matches the ``n.`` prefix in filenames).
        method_name: Short method label, e.g. ``"direct_llm"``, ``"static_workflow"``, ``"mas"``.
        n_fields: Number of fields in the standard used (embedded in the filename for traceability).
        base_dir: Base output directory (default: project-level ``outputs/``).
        date: Date string ``YYYY-MM-DD``; defaults to today.

    Returns:
        Full path to the (not-yet-created) CSV file.

    Example:
        >>> path = get_method_output_path(1, "direct_llm", 42)
        >>> print(path)
        'outputs/2026-03-11/1_direct_llm_42fields_2026-03-11.csv'
    """
    filename = get_timestamped_filename(
        f"{paper_index}_{method_name}_{n_fields}fields",
        extension="csv",
        include_time=False,
    )
    return get_output_path(filename, base_dir=base_dir, date=date)


def _normalize_cell_value(value: Any) -> str:
    """Normalize a Python value into a CSV cell string."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def save_json_records_to_csv(
    json_input: Any,
    path: Union[str, Path],
    record_key: str = "yield_records",
) -> int:
    """
    Parse JSON records from a variety of input formats and save them to a CSV file.

    Accepts:
    - A list of record dicts
    - A dict with a records key (``record_key``, ``"yield_records"``, ``"records"``,
      ``"data"``, or ``"final_meta_analysis_records"``)
    - A JSON string (with optional markdown code-block fences) containing either of
      the above structures

    Behavior:
    - If the file does not exist or is empty, writes header + rows.
    - If the file exists and is non-empty, verifies that the existing header matches
      the new records. If they match, appends rows; otherwise raises ``ValueError``.

    Args:
        json_input: Records as a list, dict, or JSON string.
        path: Destination CSV file path.
        record_key: Key used to locate the records list inside a dict or parsed JSON.

    Returns:
        Number of records written.

    Raises:
        ValueError: If the existing CSV header does not match the new records' schema.
    """
    _COMMON_KEYS = [record_key, "yield_records", "records", "data",
                    "final_meta_analysis_records"]

    def _strip_fences(text: str) -> str:
        text = text.strip()
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'^```\s*',     '', text)
        text = re.sub(r'\s*```$',     '', text)
        return text.strip()

    def _extract_from_parsed(parsed: Any) -> List[Dict]:
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for k in _COMMON_KEYS:
                if k in parsed and isinstance(parsed[k], list):
                    return parsed[k]
            return [parsed]
        return []

    # ── Resolve records ──────────────────────────────────────────────────────
    if isinstance(json_input, list):
        records = json_input

    elif isinstance(json_input, dict):
        # Try the caller-specified key first, then common fallbacks
        for k in _COMMON_KEYS:
            if k not in json_input:
                continue
            val = json_input[k]
            if isinstance(val, list):
                records = val
                break
            if isinstance(val, str):
                records = _extract_from_parsed(json.loads(_strip_fences(val)))
                break
        else:
            records = [json_input]

    else:
        # String / bytes
        records = _extract_from_parsed(json.loads(_strip_fences(str(json_input))))

    # ── Flatten any accidentally nested lists and drop non-dict items ────────
    flat: List = []
    for item in records:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    records = [r for r in flat if isinstance(r, dict)]

    if not records:
        return 0

    fieldnames = sorted({key for rec in records for key in rec.keys()})
    csv_path   = str(path)
    file_empty = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0

    if file_empty:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                writer.writerow({fn: _normalize_cell_value(rec.get(fn))
                                  for fn in fieldnames})
        return len(records)

    # Existing file: verify header compatibility before appending
    with open(csv_path, newline="", encoding="utf-8") as f:
        existing_header = next(csv.reader(f), [])

    if existing_header != fieldnames:
        only_existing = sorted(set(existing_header) - set(fieldnames))
        only_new      = sorted(set(fieldnames)      - set(existing_header))
        raise ValueError(
            "CSV schema mismatch.\n"
            f"  Existing header : {existing_header}\n"
            f"  New header      : {fieldnames}\n"
            f"  Only in existing: {only_existing}\n"
            f"  Only in new     : {only_new}"
        )

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=existing_header)
        for rec in records:
            writer.writerow({fn: _normalize_cell_value(rec.get(fn))
                              for fn in existing_header})

    return len(records)


# Export all functions
__all__ = [
    "get_dated_output_dir",
    "get_output_path",
    "save_records_to_csv",
    "save_extraction_results",
    "save_extraction_results_with_timestamp",
    "get_timestamped_filename",
    "list_output_files",
    "clean_old_outputs",
    "DEFAULT_OUTPUT_DIR",
    # Method output path helper
    "get_method_output_path",
    # JSON → CSV utilities
    "_normalize_cell_value",
    "save_json_records_to_csv",
]
