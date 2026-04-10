"""
Static (non-agent) workflows for LLM-powered processing.

- **facts** — text → SPO facts → optional facts→schema LLM.
- **label_then_direct** — LLM outputs full tagged paper text → direct-LLM-style record extraction.
"""

from .two_step_text_to_dataset import (
    DirectPromptStyle,
    Fact,
    FactExtractionResult,
    FieldValuePairEntry,
    FieldValuePairsExtraction,
    LabeledPaperOutput,
    WorkflowMode,
    extract_facts_from_text,
    extract_field_value_pairs_from_text,
    label_text_with_field_pairs,
    llm_build_dataset_from_facts,
    llm_extract_records_from_tagged_paper,
    llm_produce_labeled_paper_text,
    pairs_extraction_to_tuples,
    build_fact_dataset,
    build_dataset_from_schema_output,
    validate_fact_dataset,
    validate_schema_records_dataset,
    run_two_step_text_to_dataset,
)

__all__ = [
    "DirectPromptStyle",
    "Fact",
    "FactExtractionResult",
    "FieldValuePairEntry",
    "FieldValuePairsExtraction",
    "LabeledPaperOutput",
    "WorkflowMode",
    "extract_facts_from_text",
    "extract_field_value_pairs_from_text",
    "label_text_with_field_pairs",
    "llm_build_dataset_from_facts",
    "llm_extract_records_from_tagged_paper",
    "llm_produce_labeled_paper_text",
    "pairs_extraction_to_tuples",
    "build_fact_dataset",
    "build_dataset_from_schema_output",
    "validate_fact_dataset",
    "validate_schema_records_dataset",
    "run_two_step_text_to_dataset",
]
