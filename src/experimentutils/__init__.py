"""
Experiment utilities for meta-analysis experiments.

This module provides utility functions for:
- Finding and listing paper markdown files
- Managing experiment data paths
- Saving results with date-organized outputs
- Batch processing utilities
"""
from .eval_utils import (
    load_ground_truth,
    build_study_paper_mapping,
    get_paper_path_for_study,
    highlight_numbers_and_tables,
    build_extraction_context,
)
from .file_utils import (
    get_all_markdown_paths,
    get_all_paper_paths,
    read_paper_text,
    convert_pdf_to_markdown,
    convert_all_pdfs_to_markdown,
    get_markdown_paths_from_directory,
    get_paper_info_from_path,
    list_paper_folders,
    get_paper_count,
    filter_paths_by_pattern,
)

from .output_utils import (
    get_dated_output_dir,
    get_output_path,
    save_records_to_csv,
    save_extraction_results,
    save_extraction_results_with_timestamp,
    get_timestamped_filename,
    list_output_files,
    clean_old_outputs,
    save_json_records_to_csv,
    get_method_output_path,
)

from .standard_utils import (
    filter_standard,
    get_metadata_field_keys,
)

from .progress_utils import ProgressOrchestrator


__all__ = [
    # File utilities
    "get_all_markdown_paths",
    "get_all_paper_paths",
    "read_paper_text",
    "convert_pdf_to_markdown",
    "convert_all_pdfs_to_markdown",
    "get_markdown_paths_from_directory",
    "get_paper_info_from_path",
    "list_paper_folders",
    "get_paper_count",
    "filter_paths_by_pattern",
    # Output utilities
    "get_dated_output_dir",
    "get_output_path",
    "save_records_to_csv",
    "save_extraction_results",
    "save_extraction_results_with_timestamp",
    "get_timestamped_filename",
    "list_output_files",
    "clean_old_outputs",
    "save_json_records_to_csv",
    "get_method_output_path",
    # Standard utilities
    "filter_standard",
    "get_metadata_field_keys",
    # Progress utilities
    "ProgressOrchestrator",
    # Evaluation utilities
    "load_ground_truth",
    "build_study_paper_mapping",
    "get_paper_path_for_study",
    "highlight_numbers_and_tables",
    "build_extraction_context",
]
