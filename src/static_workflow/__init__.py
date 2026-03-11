"""
Static (non-agent) workflows for LLM-powered processing.

This package contains small, composable pipelines that mirror
multi-agent systems, but are implemented as simple, deterministic
function chains.

Currently available:

- Two-step text → facts → dataset workflow
"""

from .two_step_text_to_dataset import (
    Fact,
    FactExtractionResult,
    extract_facts_from_text,
    build_fact_dataset,
    llm_build_dataset_from_facts,
    build_dataset_from_schema_output,
    validate_fact_dataset,
    run_two_step_text_to_dataset,
)

__all__ = [
    "Fact",
    "FactExtractionResult",
    "extract_facts_from_text",
    "build_fact_dataset",
    "llm_build_dataset_from_facts",
    "build_dataset_from_schema_output",
    "validate_fact_dataset",
    "run_two_step_text_to_dataset",
]

