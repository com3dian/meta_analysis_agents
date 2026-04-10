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
META_ANALYSIS_EXTRACTION_PROMPT_TEMPLATE = """You are an expert agricultural data extraction specialist. Your task is to extract **measured crop yield information** and related agronomic/context variables from scientific research papers. **Maximize recall:** your output list should include **every** distinct experiment-level or table-level data row the paper supports, not a summary.

**MULTI-STEP REASONING PROCESS**

### STEP 1: Anchor Identification
Scan the **entire** content (paragraphs, **all** tables row-by-row, captions, footnotes, supplements) and locate **every** sentence or cell that reports **actual measured** yields, densities, inputs, or other schema fields. Ignore model metrics (RMSE, R²), yield *gaps*, and purely simulated outputs.

### STEP 2: Contextual Reasoning
For **each** anchor (each table cell or sentence that could become one row), gather context: year, treatment, fertilization level, cropping system (sole vs intercrop), species, and block/plot if given.

### STEP 3: Completeness, Evidence & Confidence
If any field for that row is missing, search these locations before leaving null:
- **Year / season / duration** → Abstract, Methods, table row labels, column headers
- **Location / coordinates** → Site description, Methods, tables
- **Crops / species / intercrop pattern** → Abstract, Methods, table/figure labels
- **Treatments (water, NPK, etc.)** → Experimental design, Methods, **table columns**
- **Densities, inputs, nutrients** → Methods and **numeric table cells**
- **Each yield column** (e.g. sole crop 1, sole crop 2, intercrop 1, intercrop 2) → map to the matching schema fields **per row**

Assign confidence (high/medium/low) mentally; still **output the row** if values are supported.

### STEP 4: Record Construction — one row per experimental observation
Build **one schema record per distinct observation**:
- **Each row** of a main results table (year × treatment × system) → typically **one record** (or more if the table splits species into separate logical rows).
- **Intercropping:** if the table has separate columns for species 1 vs 2 vs intercrop yields, **fill all applicable yield fields** for that row; do not merge multiple years into one record.
- **Factorial / split-plot:** each combination of factors that has its own numeric result → **separate record** when the schema expects trial-level rows.
- Keep original units; **do not convert**. Use `null` only when the paper never reports that field for that row.

**De-duplication:** Remove duplicates **only** when two records would be identical on the same year, same treatment level, same cropping system, and same species context. When in doubt, **keep both** or prefer **more records**.

**TARGET DATA TO EXTRACT**:
- ✅ INCLUDE: Field-measured yields, dry matter, densities, nutrient applications as in the schema
- ❌ EXCLUDE: Model evaluation metrics (RMSE, R², MAE), predictions, correlation-only statistics

**IMPORTANT NOTES (recall)**:
- **Extract ALL rows** implied by the main results tables — the count of records should usually be **similar to** (or exceed) the number of distinct table rows × relevant treatment levels, not a single aggregate record.
- **Intercropping:** separate records or fully filled columns per table row for `unified yield sc 1`, `sc 2`, `ic 1`, `ic 2` (or equivalent) when data exist.
- Dry matter in g m⁻², kg ha⁻¹, etc. are valid.

**META-ANALYTIC SCHEMA**:
{schema}

Now analyze the following paper and extract all records following the schema:

---

{document_content}

---

Extract **all** records from this paper following the schema. Prefer **too many** distinct records over too few."""


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


TAGGED_DOCUMENT_PROMPT_PREFIX = """The paper below includes XML: <FieldName>verbatim text</FieldName> where FieldName matches a schema field. Tags mark evidence — use them first, then **every** table row and caption in the tagged text to maximize how many complete records you output. Missing tags does not mean missing data: still read full tables.

"""

# Appended to step-2 (records) prompts when maximum recall is desired (static workflow label_then_direct, etc.).
RECORD_EXTRACTION_COMPLETENESS_BLOCK = """
**OUTPUT COMPLETENESS — maximize record count (mandatory)**:
- **Goal:** The output records list should contain **as many records** as the paper’s evidence supports — typically **one record per distinct row** in main yield/results tables (each year × treatment × sole/intercrop combination that has its own numbers).
- **Intercropping / multi-column tables:** For each table row, populate **all** yield columns the schema has (e.g. species 1, species 2, intercrop components) from tags or adjacent cells — **do not** collapse the whole table into one or two summary records.
- **Do not merge** rows that differ by year, N level, or treatment unless the schema explicitly describes one aggregate row.
- Scan the **entire** tagged document: Methods + Results + every table; if a second table adds years or treatments, add more records.
- Use `null` only when that cell/field is truly absent after a full pass — not because you summarized multiple rows into one.
- If unsure between **one wide record** vs **several narrower records**, prefer **several** records that match table structure.
"""


def append_record_completeness_instructions(prompt: str) -> str:
    """Append :data:`RECORD_EXTRACTION_COMPLETENESS_BLOCK` to a record-extraction prompt."""
    return prompt.rstrip() + "\n\n" + RECORD_EXTRACTION_COMPLETENESS_BLOCK


def get_tagged_extraction_prompt(
    document_content: str,
    schema: Union[str, Dict[str, Any]],
    *,
    include_tag_note: bool = True,
) -> str:
    """
    Same core prompt as :func:`get_extraction_prompt`, optionally prefixed with a short
    note about schema-aligned XML tags (for labeller → extractor pipelines).

    Set ``include_tag_note=False`` to match a plain direct LLM call exactly while still
    passing tagged ``document_content``.
    """
    body = get_extraction_prompt(document_content, schema)
    if not include_tag_note:
        return body
    return TAGGED_DOCUMENT_PROMPT_PREFIX + body


def get_tagged_simple_extraction_prompt(
    document_content: str,
    schema: Union[str, Dict[str, Any]],
    *,
    include_tag_note: bool = True,
) -> str:
    """Same as :func:`get_simple_extraction_prompt` with optional XML-tag note prefix."""
    body = get_simple_extraction_prompt(document_content, schema)
    if not include_tag_note:
        return body
    return TAGGED_DOCUMENT_PROMPT_PREFIX + body


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
    
    return f"""Extract crop yield and related trial data from the following research paper.

Schema:
{schema_str}

Rules (maximize completeness):
- **One record per distinct table row / trial combination** (year × treatment × cropping system) when numbers differ — do not output a single summary row for the whole paper.
- Extract **all** such records from tables and text; intercropping: fill each species/yield column when present.
- Include units with values; set fields to null only if not reported.
- Prefer **more records** over merged summaries.

Paper content:
---
{document_content}
---

Return all extracted records."""


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
    "TAGGED_DOCUMENT_PROMPT_PREFIX",
    "RECORD_EXTRACTION_COMPLETENESS_BLOCK",
    "append_record_completeness_instructions",
    "get_extraction_prompt",
    "get_tagged_extraction_prompt",
    "get_tagged_simple_extraction_prompt",
    "get_simple_extraction_prompt",
    "get_custom_extraction_prompt",
]
