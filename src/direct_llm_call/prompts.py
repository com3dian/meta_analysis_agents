"""
Prompt templates for direct LLM extraction of meta-analysis data.

This module contains the extraction prompt that instructs the LLM to extract
crop yield information and related agronomic variables from research papers.

The prompts are designed to work with dynamic schema definitions from
METADATA_STANDARDS, allowing flexible extraction of different data types.
"""
from typing import Any, Dict, Union

from .schemas import format_schema_for_prompt


# The base extraction prompt template
# Uses {schema} and {document_content} placeholders
META_ANALYSIS_EXTRACTION_PROMPT_TEMPLATE = """You are an expert agricultural data extraction specialist. Your task is to extract **measured crop yield information** and related agronomic/context variables from scientific research papers. You must reason carefully and follow a multi-step extraction process to ensure accurate and complete data.

**MULTI-STEP REASONING PROCESS**

### STEP 1: Anchor Identification
Scan the entire content (including paragraphs, tables, captions, footnotes, and supplements) to locate all sentences or table entries that mention **actual measured crop yield values** (e.g., "maize yield was 7.2 t/ha", "cassava storage root dry matter was 1357 g m⁻²"). Ignore model performance metrics (e.g., RMSE, R²), yield gaps, and simulated/modelled outputs.

### STEP 2: Contextual Reasoning
For each identified yield anchor, analyze its surrounding context (paragraphs, section headers, tables, captions) to extract the associated fields as defined in the schema below.

If the fields are explicitly stated in the anchor or nearby context, extract them directly.

### STEP 3: Completeness, Evidence & Confidence
If any required field for a yield record is **missing**, try to retrieve it from specific related sections:
- For missing **location** or **year**, check **Materials and Methods / Site description**.
- For missing **crop type**, check **Abstract** and **Materials and Methods**.
- For missing **treatment (water/fertilizer)**, check **Field management / Experimental design / Plot / Block / Treatment / Table**.
- For missing **tillage (density & cultivation)**, check **Planting / Agronomic practices / Cultivation / Management**.
- For missing **soil properties**, check **Soil / Site description / Materials and Methods / Table**.
- For missing **climate**, check **Climate / Weather / Meteorology / Data sources**.
- For missing **UAV data**, check **UAV / UAS / Flight / Sensor / Data acquisition / Table / Figure captions**.

Assign confidence levels based on:
- **high**: Value is explicitly stated with clear context
- **medium**: Value is inferred from context or partially stated
- **low**: Value is uncertain or requires significant interpretation

### STEP 4: Record Construction
For each complete and validated record, construct a structured entry following the schema. Keep original units and expressions; **do not convert**. If a field is not found with reasonable certainty, set its value to `null`. Apply de-duplication across records where key identifiers are identical.

**TARGET DATA TO EXTRACT**:
- ✅ INCLUDE: Actual field-measured or reported yields (e.g., from experiments, harvest trials, dry matter accumulation)
- ❌ EXCLUDE: Model evaluation metrics (RMSE, R², MAE), yield gaps, predictions, correlation coefficients

**IMPORTANT NOTES**:
- Extract ALL yield records found in the paper, not just one
- Each unique combination of crop type, treatment, year, and location should be a separate record
- For intercropping studies, extract yields for each crop component separately
- Dry matter production values (e.g., g m⁻², kg ha⁻¹) are valid yield measurements
- Storage root dry matter, pod dry matter, etc. are valid yield components

**META-ANALYTIC SCHEMA**:
{schema}

Now analyze the following paper and extract all records following the schema:

---

{document_content}

---

Extract all records from this paper following the schema provided."""


def get_extraction_prompt(
    document_content: str,
    schema: Union[str, Dict[str, Any]]
) -> str:
    """
    Format the extraction prompt with the document content and schema.
    
    Args:
        document_content: The text content of the research paper (markdown format)
        schema: The metadata standard schema (from METADATA_STANDARDS or custom dict)
        
    Returns:
        The formatted prompt string ready for LLM invocation
        
    Example:
        >>> from src.standards import METADATA_STANDARDS
        >>> 
        >>> prompt = get_extraction_prompt(
        ...     document_content=paper_text,
        ...     schema=METADATA_STANDARDS["climate_vs_cropyield"]
        ... )
    """
    schema_str = format_schema_for_prompt(schema)
    return META_ANALYSIS_EXTRACTION_PROMPT_TEMPLATE.format(
        schema=schema_str,
        document_content=document_content
    )


def get_simple_extraction_prompt(
    document_content: str,
    schema: Union[str, Dict[str, Any]]
) -> str:
    """
    Get a simpler, more concise extraction prompt.
    
    This version is shorter and may work better with smaller models
    or when token limits are a concern.
    
    Args:
        document_content: The text content of the research paper
        schema: The metadata standard schema
        
    Returns:
        The formatted prompt string
    """
    schema_str = format_schema_for_prompt(schema)
    
    return f"""Extract crop yield data from the following research paper.

For each yield measurement found, create a record following this schema:
{schema_str}

Rules:
- Extract ALL yield records (not just one)
- Include units with values (e.g., "8.5 t/ha")
- Set fields to null if not found
- Include confidence levels: high/medium/low

Paper content:
---
{document_content}
---

Return the extracted records."""


def get_custom_extraction_prompt(
    document_content: str,
    schema: Union[str, Dict[str, Any]],
    custom_instructions: str = ""
) -> str:
    """
    Create a custom extraction prompt with additional instructions.
    
    Args:
        document_content: The text content of the research paper
        schema: The metadata standard schema
        custom_instructions: Additional instructions to include in the prompt
        
    Returns:
        The formatted prompt string with custom instructions
    """
    schema_str = format_schema_for_prompt(schema)
    
    base_prompt = f"""You are an expert data extraction specialist. Extract structured data from the following research paper.

**SCHEMA**:
{schema_str}

**INSTRUCTIONS**:
- Extract all matching records from the paper
- Follow the schema exactly
- Set fields to null if not found with reasonable certainty
- Keep original units and expressions
"""
    
    if custom_instructions:
        base_prompt += f"\n**ADDITIONAL INSTRUCTIONS**:\n{custom_instructions}\n"
    
    base_prompt += f"""
**PAPER CONTENT**:
---
{document_content}
---

Extract all records following the schema."""
    
    return base_prompt


# Export prompt-related utilities
__all__ = [
    "META_ANALYSIS_EXTRACTION_PROMPT_TEMPLATE",
    "get_extraction_prompt",
    "get_simple_extraction_prompt", 
    "get_custom_extraction_prompt",
]
