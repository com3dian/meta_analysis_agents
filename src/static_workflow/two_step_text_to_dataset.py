"""
Static workflows over paper text:

**``workflow=\"facts\"`` (default)** — SPO facts → optional facts→schema LLM.

**``workflow=\"label_then_direct\"``** — (1) Single LLM call outputs the **full paper text**
with inline ``<FieldName>value</FieldName>`` tags (no separate tagging tool). (2) Second LLM
uses the direct-LLM extraction prompt family plus optional **completeness** instructions.
"""

import json
import logging
import re
from typing import List, Tuple, Optional, Dict, Any, Union, Type, Literal

import pandas as pd
from pydantic import BaseModel, Field

from src.direct_llm_call.prompts import (
    append_record_completeness_instructions,
    get_extraction_prompt,
    get_simple_extraction_prompt,
    get_tagged_extraction_prompt,
    get_tagged_simple_extraction_prompt,
)
from src.direct_llm_call.utils import invoke_llm_with_structured_output, invoke_with_schema
from src.core.schema_factory import create_output_schema
from src.tools.xml_tagging import apply_xml_tags_to_content

logger = logging.getLogger(__name__)


class Fact(BaseModel):
    """
    Single extracted fact in (subject, predicate, object) form.
    """

    subject: str = Field(
        ...,
        description=(
            "Main entity: crop, treatment, site, variable, table row label, or design element."
        ),
        examples=["Cassava", "Nitrogen fertilizer", "Intercropping system"],
    )
    predicate: str = Field(
        ...,
        description="Relationship, measurement type, or property (e.g. 'yield was', 'applied at', 'p-value').",
        examples=["increases yield", "reduces weed biomass"],
    )
    object: str = Field(
        ...,
        description="Value with units if any, comparison arm, or target entity.",
        examples=["by 20%", "soil nitrogen content", "cowpea"],
    )
    source_span: Optional[str] = Field(
        None,
        description="Short verbatim quote or sentence from the document supporting this fact.",
    )


class FactExtractionResult(BaseModel):
    """
    Wrapper model for structured LLM output.
    """

    facts: List[Fact] = Field(
        default_factory=list,
        description="List of extracted facts from the input document.",
    )


class FieldValuePairEntry(BaseModel):
    """One schema field value anchored in the document (for XML tagging)."""

    field_name: str = Field(
        ...,
        description="Exact field name from the provided schema.",
    )
    value_text: str = Field(
        ...,
        description="Exact text span from the document; copy verbatim for tagging.",
    )


class FieldValuePairsExtraction(BaseModel):
    """Structured output for pair-based labelling (optional; not used by default ``label_then_direct``)."""

    field_value_pairs: List[FieldValuePairEntry] = Field(
        default_factory=list,
        description="List of (field, value) pairs to tag in the document.",
    )


class LabeledPaperOutput(BaseModel):
    """
    Step-1 labeller output: the **entire** paper as one string with inline XML tags.
    No deterministic tagging tool — the model returns tagged text directly.
    """

    labeled_document: str = Field(
        ...,
        description=(
            "Full document text with inline <SchemaFieldName>verbatim span</SchemaFieldName> "
            "for each schema-relevant value found. Must preserve the source text; do not summarize."
        ),
    )


WorkflowMode = Literal["facts", "label_then_direct"]


FIELD_PAIRS_PROMPT_TEMPLATE = """You are a schema-aware value extraction specialist.

Scan the ENTIRE document and list concrete values for the fields defined below.
Use ONLY these field names (exact spelling). Copy each value EXACTLY as it appears
in the text (including units).

**Schema fields (name — description):**
{schema_descriptions}

**Document:**
\"\"\"
{text}
\"\"\"

**Rules:**
- Return at most {max_pairs} pairs; prioritise values needed for tables/results (yields, densities, dates, design, nutrients, species).
- List EACH table row occurrence separately — do NOT collapse multiple rows into one pair.
- Skip fields with no evidence in the document.
- Do not paraphrase; `value_text` must be a substring the tagging step can find in the document.

Return ONLY structured data according to the provided schema.
"""


LABELLER_DOCUMENT_PROMPT_TEMPLATE = """You are a document labeller for downstream structured extraction.

**Task:** Produce a single field `labeled_document`: the **complete** input document, **reproduced in full**, with inline XML so step 2 can emit **one output record per experimental row** where possible.

**Schema fields — tag names must match EXACTLY (use these as XML element names):**
{schema_descriptions}

**Tagging rules (maximize coverage for recall):**
- Preserve the document **verbatim** — no shortening, no summarizing, no skipped tables.
- Wrap **every** schema-relevant number or label as `<FieldName>exact substring from source</FieldName>`.
- **Tables:** tag **each cell** that maps to a field — especially **each row** of yield/density/input tables (same field name may appear many times for different rows). If a table has columns for species 1, species 2, intercrop yields, tag **each numeric cell** with the correct field name.
- **Years, treatments, N levels:** tag distinct values so rows can be split downstream (e.g. year in each row, treatment labels).
- Do not invent text; only wrap spans that exist.
- Keep markdown/table structure.

**Why:** Sparse tags → few records downstream. **Dense, row-level tags** → complete extraction.

**Input document:**
\"\"\"
{text}
\"\"\"

Return ONLY structured data: one object with `labeled_document` = full tagged text.
"""


FACT_PROMPT_TEMPLATE = """You are an information extraction assistant for research papers (agronomy, ecology, meta-analysis-friendly).

Read the **entire** document below and extract **atomic** facts as (subject, predicate, object) triples.
Optimize for **breadth and faithfulness**: capture as much *usable* information as the fact budget allows.

**Cover (where present in the text):**
- Study context: crop(s), location, site, season, years, experimental design, replication
- Treatments: levels, doses, timing, cultivars, intercrops, tillage, irrigation, inputs
- Methods: plot size, sampling, statistical models, tests
- Results: means, SE/SD, yields, densities, biomass, % change, CI, p-values — prefer **separate** facts for distinct table rows, treatment levels, or non-redundant numeric results
- Environmental / soil / climate / nutrient details when explicitly stated

**Rules:**
- Return **up to {max_facts}** facts; use the full budget with **distinct** items (do not stop early after a handful of generic summaries).
- Each fact must be grounded in the text; **do not invent**.
- Keep each fact **one atomic claim**; split combined sentences if they encode multiple values.
- **source_span**: short verbatim quote or sentence supporting the fact (strongly preferred for numeric results).
- For **tables** (especially yield, density, inputs): emit **one fact per table row or per distinct (year × treatment × system)** numeric outcome — do not replace six table cells with one summary fact.

{schema_hint_section}
Document:
\"\"\"{text}\"\"\"

Return ONLY structured data according to the provided schema.
"""


def extract_facts_from_text(
    text: str,
    max_facts: int = 200,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    provider: Optional[str] = None,
    *,
    schema_standard: Optional[Union[str, Dict[str, Any]]] = None,
) -> FactExtractionResult:
    """
    Step 1: Fact Extractor (Text → Facts).

    Uses a structured-output LLM call to convert raw text into a list of `Fact`.
    This is analogous to a first agent whose sole responsibility is extraction.

    Args:
        text: Raw input document text.
        max_facts: Maximum number of facts to extract (default 200 for dense tables).
        model_name: Optional LLM model override.
        temperature: LLM temperature (0.0 for deterministic output).
        provider: Optional provider override (google, openai, surf, qwen).
        schema_standard: When set (e.g. same dict/JSON as the dataset builder), the prompt
            includes target field names so facts align with multi-row schema extraction.

    Returns:
        FactExtractionResult containing a list of facts.
    """
    logger.info(
        "extract_facts_from_text: starting (text_len=%d, max_facts=%d, schema_hint=%s)",
        len(text),
        max_facts,
        schema_standard is not None,
    )
    hint = _schema_hint_section_for_facts(schema_standard) if schema_standard else ""
    prompt = FACT_PROMPT_TEMPLATE.format(
        text=text,
        max_facts=max_facts,
        schema_hint_section=hint,
    )
    logger.debug("extract_facts_from_text: prompt_len=%d", len(prompt))

    result = invoke_llm_with_structured_output(
        prompt=prompt,
        output_schema=FactExtractionResult,
        model_name=model_name,
        temperature=temperature,
        provider=provider,
        records_key="facts",
    )
    logger.info(
        "extract_facts_from_text: completed (extracted %d facts)",
        len(result.facts),
    )
    return result


def build_fact_dataset(
    facts_or_result: FactExtractionResult | List[Fact],
) -> pd.DataFrame:
    """
    Step 2a: Dataset Builder (Facts → DataFrame).

    Converts a list of `Fact` objects (or a `FactExtractionResult`) into
    a tidy pandas DataFrame suitable for analysis or export.

    Columns:
        - subject
        - predicate
        - object
        - source_span
    """
    if isinstance(facts_or_result, FactExtractionResult):
        facts = facts_or_result.facts
    else:
        facts = facts_or_result

    records = [fact.model_dump() for fact in facts]
    df = pd.DataFrame(records)
    logger.debug("build_fact_dataset: built DataFrame with %d rows", len(df))
    return df


def validate_fact_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Step 2b: Dataset Validator (DataFrame → (ok, issues)).

    Runs a few lightweight checks to ensure the dataset is usable:
    - DataFrame is not empty
    - Required columns are present
    - No completely empty subject/predicate/object rows

    Returns:
        (ok, issues)
        - ok: True if dataset passes all checks
        - issues: list of human-readable issue descriptions
    """
    issues: List[str] = []

    if df.empty:
        issues.append("Dataset is empty (no facts extracted).")

    required_cols = ["subject", "predicate", "object"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")
    else:
        for col in required_cols:
            if df[col].isna().all():
                issues.append(f"Column '{col}' is entirely empty.")

    ok = len(issues) == 0
    logger.debug(
        "validate_fact_dataset: ok=%s, issues=%s",
        ok,
        issues,
    )
    return ok, issues


def validate_schema_records_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Checks after facts→schema LLM path (DataFrame is schema rows, not SPO columns)."""
    issues: List[str] = []
    if df.empty:
        issues.append("Dataset is empty (no records extracted).")
    return len(issues) == 0, issues


DATASET_BUILDER_PROMPT_TEMPLATE = """You are a dataset builder for structured extraction.

You are given **extracted facts** (subject, predicate, object, source_span) and a **target schema**.
Your job is to build the **largest** set of valid schema rows the paper supports — **match or approach** the number of distinct trial/table rows in the Results, not a one-row summary.

{document_section}

**Target schema (field names and descriptions):**
{schema_descriptions}

**Extracted facts (JSON):**
{facts_json}

**Rules:**
- Use **all** relevant facts; never drop evidence to simplify.
- Produce **one record per** distinct experimental observation: typically **each row** of a main results table = **one record** when yields/treatments differ.
- Map (subject, predicate, object) to fields; merge facts only when they clearly describe the **same** table row.
- Prefer **exact strings** from facts or document.
- **Do not invent** values; use null when unsupported.
{fill_rules}

**Multi-record rule (CRITICAL):**
- **Tables:** If Results Table 3 has 8 rows of yields across years and treatments, aim for **about 8 records** (or one per schema-defined split), not 1–2 merged records.
- **Intercropping:** Fill `unified yield sc 1`, `sc 2`, `ic 1`, `ic 2` (or equivalent) from the correct columns **per row**; repeat site/design fields on each row.
- Facts are a **hint list** — if the document excerpt shows **more rows** than facts, **emit records from the excerpt** up to what the schema allows.
- **Never** put comma-separated years in a single `Year` field when the table has **separate rows per year** — use **separate records**.

**Output shape (CRITICAL):** Return one JSON object whose ONLY top-level key is `"{records_key}"`,
and whose value is the array of record objects. Do not use `records`, `data`, or a bare array.

Return ONLY structured data according to the provided schema.
"""

DATASET_BUILDER_FILL_RULES_FACTS_ONLY = """
- If a field cannot be inferred from the facts, set it to null.
"""

DATASET_BUILDER_FILL_RULES_WITH_DOCUMENT = """
- When a field cannot be inferred from the facts alone, search the **full document excerpt** below (Methods, Results, tables, figure captions, supplementary-style blocks).
- **Prioritize Results tables and yield/nutrient panels:** read row labels and column headers to assign values to the correct schema fields (e.g. sole crop vs intercrop, species 1 vs 2).
- If you find a supported value in the document, use it. If the paper states it was not reported, use "Not explicitly stated".
- If the field is truly absent from both facts and excerpt, set it to null.
- Prefer **more records** when the text supports multiple distinct yield/treatment rows rather than collapsing them.
- If the excerpt shows **more** data rows than the facts list, still emit **one record per evident row** using the excerpt; facts are hints, not a hard ceiling on row count.
"""


def _parse_schema_to_dict(standard: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Parse schema standard (JSON string or dict) into a field -> description dict."""
    if isinstance(standard, dict):
        return standard
    cleaned = standard.strip()
    if cleaned.startswith("```json"):
        cleaned = re.sub(r"^```json\s*", "", cleaned)
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\s*", "", cleaned)
    if cleaned.endswith("```"):
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned.strip())


def _format_schema_descriptions(standard: Union[str, Dict[str, Any]]) -> str:
    """Format schema field descriptions for inclusion in the dataset builder prompt."""
    schema_dict = _parse_schema_to_dict(standard)
    lines = []
    for field_name, desc in schema_dict.items():
        desc_str = desc if isinstance(desc, str) else str(desc)
        lines.append(f"- **{field_name}**: {desc_str}")
    return "\n".join(lines)


def extract_field_value_pairs_from_text(
    text: str,
    dataset_standard: Union[str, Dict[str, Any]],
    *,
    max_pairs: int = 300,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    provider: Optional[str] = None,
) -> FieldValuePairsExtraction:
    """
    Labeller step: document → (field_name, value_text) pairs for XML tagging.

    Values must appear verbatim in ``text`` so :func:`apply_xml_tags_to_content` can wrap them.
    """
    logger.info(
        "extract_field_value_pairs_from_text: starting (text_len=%d, max_pairs=%d)",
        len(text),
        max_pairs,
    )
    schema_descriptions = _format_schema_descriptions(dataset_standard)
    prompt = FIELD_PAIRS_PROMPT_TEMPLATE.format(
        text=text,
        schema_descriptions=schema_descriptions,
        max_pairs=max_pairs,
    )
    result = invoke_llm_with_structured_output(
        prompt=prompt,
        output_schema=FieldValuePairsExtraction,
        model_name=model_name,
        temperature=temperature,
        provider=provider,
        records_key=None,
    )
    logger.info(
        "extract_field_value_pairs_from_text: completed (%d pairs)",
        len(result.field_value_pairs),
    )
    return result


def pairs_extraction_to_tuples(
    extraction: FieldValuePairsExtraction,
) -> List[Tuple[str, str]]:
    """Convert extraction model to (field, value) tuples for :func:`apply_xml_tags_to_content`."""
    out: List[Tuple[str, str]] = []
    for p in extraction.field_value_pairs:
        fn = (p.field_name or "").strip()
        vt = (p.value_text or "").strip()
        if fn and vt:
            out.append((fn, vt))
    return out


def label_text_with_field_pairs(
    text: str,
    pairs: List[Tuple[str, str]],
) -> str:
    """Wrap each value occurrence with ``<field>value</field>`` (MAS labeller logic)."""
    return apply_xml_tags_to_content(text, pairs)


def llm_produce_labeled_paper_text(
    text: str,
    dataset_standard: Union[str, Dict[str, Any]],
    *,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    provider: Optional[str] = None,
) -> LabeledPaperOutput:
    """
    Step 1 for ``label_then_direct``: a **single** structured LLM call whose output is only
    ``labeled_document`` — the full paper text with inline ``<FieldName>…</FieldName>`` tags.

    No intermediate (field, value) list and no :func:`apply_xml_tags_to_content` tool; the model
    returns the tagged document directly.
    """
    logger.info(
        "llm_produce_labeled_paper_text: starting (text_len=%d)",
        len(text),
    )
    schema_descriptions = _format_schema_descriptions(dataset_standard)
    prompt = LABELLER_DOCUMENT_PROMPT_TEMPLATE.format(
        text=text,
        schema_descriptions=schema_descriptions,
    )
    result = invoke_llm_with_structured_output(
        prompt=prompt,
        output_schema=LabeledPaperOutput,
        model_name=model_name,
        temperature=temperature,
        provider=provider,
        records_key=None,
    )
    logger.info(
        "llm_produce_labeled_paper_text: done (labeled_document len=%d)",
        len(result.labeled_document or ""),
    )
    return result


def _schema_hint_section_for_facts(standard: Union[str, Dict[str, Any]], *, max_field_lines: int = 120) -> str:
    """
    Optional block for step 1: list target schema field names so facts align with downstream rows.
    """
    schema_dict = _parse_schema_to_dict(standard)
    names = list(schema_dict.keys())[:max_field_lines]
    if not names:
        return ""
    bullets = "\n".join(f"  - {n}" for n in names)
    return f"""
**Target schema fields (extract facts that will help fill these — especially table-level detail):**
{bullets}
For each **distinct** results row or (year × treatment × cropping system) outcome in tables/text, prefer **separate** facts with their own `source_span` so a later step can build **one output record per row**, not one summary record for the whole paper.
"""


def _maybe_truncate_facts_json(facts_json: str, max_chars: int = 90_000) -> str:
    if len(facts_json) <= max_chars:
        return facts_json
    return (
        facts_json[:max_chars]
        + "\n... [facts JSON truncated; rely on document excerpt below for additional rows] ...\n"
    )


def _truncate_document_for_prompt(text: str, max_chars: int = 56_000) -> str:
    """
    Truncate document for prompt limits while preserving start, middle, and end
    (abstract/intro, core results/methods, discussion/tables tail).
    """
    if len(text) <= max_chars:
        return text
    overhead = 220
    budget = max(max_chars - overhead, 3000)
    third = budget // 3
    n = len(text)
    mid_start = max(0, n // 2 - third // 2)
    mid_end = min(n, mid_start + third)
    return (
        text[:third]
        + "\n\n[... document truncated: omitted after start section ...]\n\n"
        + text[mid_start:mid_end]
        + "\n\n[... document truncated: omitted before end section ...]\n\n"
        + text[-third:]
    )


DirectPromptStyle = Literal["direct_full", "direct_simple"]


def llm_extract_records_from_tagged_paper(
    labeled_text: str,
    dataset_standard: Union[str, Dict[str, Any]],
    *,
    prompt_style: DirectPromptStyle = "direct_full",
    include_tag_note: bool = True,
    step2_maximize_completeness: bool = True,
    labeled_text_max_chars: int = 120_000,
    records_key: str = "yield_records",
    record_class_name: str = "Record",
    output_class_name: str = "DatasetOutput",
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    provider: Optional[str] = None,
) -> BaseModel:
    """
    Second step of ``label_then_direct``: run the **direct LLM** extraction prompt
    (:func:`src.direct_llm_call.prompts.get_extraction_prompt` or ``get_simple_extraction_prompt``)
    on the XML-tagged document, then structured output via :func:`invoke_with_schema`
    (same path as :func:`src.direct_llm_call.extract_meta_analysis`).

    When ``step2_maximize_completeness`` is True, appends :func:`append_record_completeness_instructions`
    so the model prioritizes emitting all supported records.
    """
    doc = labeled_text
    if len(doc) > labeled_text_max_chars:
        doc = _truncate_document_for_prompt(doc, max_chars=labeled_text_max_chars)

    if prompt_style == "direct_simple":
        if include_tag_note:
            prompt = get_tagged_simple_extraction_prompt(
                doc, dataset_standard, include_tag_note=True
            )
        else:
            prompt = get_simple_extraction_prompt(doc, dataset_standard)
    else:
        if include_tag_note:
            prompt = get_tagged_extraction_prompt(
                doc, dataset_standard, include_tag_note=True
            )
        else:
            prompt = get_extraction_prompt(doc, dataset_standard)

    if step2_maximize_completeness:
        prompt = append_record_completeness_instructions(prompt)

    logger.info(
        "llm_extract_records_from_tagged_paper: prompt_style=%s include_tag_note=%s "
        "completeness=%s prompt_len=%d",
        prompt_style,
        include_tag_note,
        step2_maximize_completeness,
        len(prompt),
    )

    result = invoke_with_schema(
        prompt=prompt,
        schema=dataset_standard,
        model_name=model_name,
        temperature=temperature,
        provider=provider,
        record_class_name=record_class_name,
        output_class_name=output_class_name,
        records_key=records_key,
    )
    nrec = len(getattr(result, records_key, []))
    logger.info("llm_extract_records_from_tagged_paper: completed (num_records=%d)", nrec)
    return result


def llm_build_dataset_from_facts(
    facts_or_result: Union[FactExtractionResult, List[Fact]],
    dataset_standard: Union[str, Dict[str, Any]],
    *,
    raw_text: Optional[str] = None,
    records_key: str = "records",
    record_class_name: str = "Record",
    output_class_name: str = "DatasetOutput",
    document_excerpt_max_chars: int = 56_000,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    provider: Optional[str] = None,
) -> BaseModel:
    """
    Optional structured LLM step: Facts → Schema-conforming dataset.

    This is useful when you want a fair comparison where the "last" LLM call
    is also regularized via structured output (like the extractor).

    When raw_text is provided, the dataset builder can search the document for
    fields not present in the facts (e.g., treatment, tillage, soil from Methods),
    improving fill rate for schema fields.

    Implementation:
    - Convert `dataset_standard` (dict or JSON string) → Pydantic model via schema factory
    - Ask the LLM to map facts into that schema (optionally using raw_text for missing fields)
    - Return an instance of the generated output schema

    Args:
        facts_or_result: `FactExtractionResult` or list of `Fact`.
        dataset_standard: Schema definition dict or JSON string (like `METADATA_STANDARDS[...]`).
        raw_text: Optional full document text. When provided, the LLM may search it to fill
            schema fields not present in the facts (e.g., Treatment, Tillage, Soil_property).
        records_key: Key name for the list of records in the output model.
        record_class_name: Name for generated record model.
        output_class_name: Name for generated wrapper output model.
        document_excerpt_max_chars: Max characters passed into the prompt from `raw_text`
            (start + middle + end windows).
        model_name: Optional LLM model override.
        temperature: LLM temperature (default 0.0).
        provider: Optional provider override.

    Returns:
        Pydantic model instance matching the generated output schema.
    """
    if isinstance(facts_or_result, FactExtractionResult):
        facts = facts_or_result.facts
    else:
        facts = facts_or_result

    logger.info(
        "llm_build_dataset_from_facts: starting (num_facts=%d, records_key=%s, has_raw_text=%s)",
        len(facts),
        records_key,
        raw_text is not None,
    )

    schema_descriptions = _format_schema_descriptions(dataset_standard)
    facts_json = json.dumps([f.model_dump() for f in facts], ensure_ascii=False, indent=2)
    facts_json = _maybe_truncate_facts_json(facts_json)
    logger.debug("llm_build_dataset_from_facts: facts_json_len=%d", len(facts_json))

    if raw_text:
        doc_excerpt = _truncate_document_for_prompt(
            raw_text, max_chars=document_excerpt_max_chars
        )
        document_section = f"""
**Source document (excerpt; use together with facts to fill every schema field you can):**
\"\"\"
{doc_excerpt}
\"\"\"
"""
        fill_rules = DATASET_BUILDER_FILL_RULES_WITH_DOCUMENT
    else:
        document_section = ""
        fill_rules = DATASET_BUILDER_FILL_RULES_FACTS_ONLY

    OutputSchema: Type[BaseModel] = create_output_schema(
        standard=dataset_standard,
        record_class_name=record_class_name,
        output_class_name=output_class_name,
        records_key=records_key,
    )

    prompt = DATASET_BUILDER_PROMPT_TEMPLATE.format(
        schema_descriptions=schema_descriptions,
        facts_json=facts_json,
        document_section=document_section,
        fill_rules=fill_rules,
        records_key=records_key,
    )

    result = invoke_llm_with_structured_output(
        prompt=prompt,
        output_schema=OutputSchema,
        model_name=model_name,
        temperature=temperature,
        provider=provider,
        records_key=records_key,
    )
    num_records = len(getattr(result, records_key, []))
    logger.info(
        "llm_build_dataset_from_facts: completed (num_records=%d)",
        num_records,
    )
    return result


def build_dataset_from_schema_output(
    schema_output: BaseModel,
    *,
    records_key: str = "records",
) -> pd.DataFrame:
    """
    Convert a schema-conforming output model into a DataFrame.

    Args:
        schema_output: Pydantic output wrapper containing `records_key` list.
        records_key: Attribute name that contains the list of records.
    """
    data = schema_output.model_dump()
    records = data.get(records_key, [])
    df = pd.DataFrame(records)
    logger.debug(
        "build_dataset_from_schema_output: built DataFrame with %d rows",
        len(df),
    )
    return df


def run_two_step_text_to_dataset(
    text: str,
    max_facts: int = 200,
    dataset_standard: Optional[Union[str, Dict[str, Any]]] = None,
    dataset_records_key: str = "records",
    use_llm_dataset_builder: bool = False,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    provider: Optional[str] = None,
    *,
    workflow: WorkflowMode = "facts",
    document_excerpt_max_chars: int = 56_000,
    record_class_name: str = "Record",
    output_class_name: str = "DatasetOutput",
    pass_schema_hint_to_fact_extractor: bool = True,
    labeled_text_max_chars: int = 120_000,
    label_step2_prompt_style: DirectPromptStyle = "direct_full",
    label_step2_include_tag_note: bool = True,
    label_step2_maximize_completeness: bool = True,
) -> Dict[str, Any]:
    """
    ``workflow=\"facts\"`` (default): SPO facts → optional facts→schema LLM.

    ``workflow=\"label_then_direct\"``: (1) :func:`llm_produce_labeled_paper_text` — one LLM call
    returns the **full** paper string with inline XML tags only (no pair list, no tagging tool).
    (2) :func:`llm_extract_records_from_tagged_paper` — direct-LLM-style prompt on that string
    via :func:`invoke_with_schema`, with optional completeness appendix.

    For ``label_then_direct``, ``dataset_standard`` is required; ``use_llm_dataset_builder`` is ignored.
    """
    logger.info(
        "run_two_step_text_to_dataset: starting workflow=%s text_len=%d use_llm=%s max_facts=%d",
        workflow,
        len(text),
        use_llm_dataset_builder,
        max_facts,
    )

    if workflow == "label_then_direct":
        if dataset_standard is None:
            raise ValueError("dataset_standard is required when workflow='label_then_direct'")

        label_step1 = llm_produce_labeled_paper_text(
            text=text,
            dataset_standard=dataset_standard,
            model_name=model_name,
            temperature=temperature,
            provider=provider,
        )
        labeled_text = (label_step1.labeled_document or "").strip()
        if not labeled_text:
            logger.warning("run_two_step_text_to_dataset: labeled_document empty from step 1")

        schema_output = llm_extract_records_from_tagged_paper(
            labeled_text=labeled_text,
            dataset_standard=dataset_standard,
            prompt_style=label_step2_prompt_style,
            include_tag_note=label_step2_include_tag_note,
            step2_maximize_completeness=label_step2_maximize_completeness,
            labeled_text_max_chars=labeled_text_max_chars,
            records_key=dataset_records_key,
            record_class_name=record_class_name,
            output_class_name=output_class_name,
            model_name=model_name,
            temperature=temperature,
            provider=provider,
        )
        df = build_dataset_from_schema_output(
            schema_output, records_key=dataset_records_key
        )
        ok, issues = validate_schema_records_dataset(df)

        logger.info(
            "run_two_step_text_to_dataset: label_then_direct done (labeled_len=%d, rows=%d)",
            len(labeled_text),
            len(df),
        )
        return {
            "workflow": workflow,
            "facts_result": None,
            "label_step1": label_step1,
            "labeled_text": labeled_text,
            "schema_output": schema_output,
            "dataset": df,
            "validation_ok": ok,
            "validation_issues": issues,
        }

    schema_hint = (
        dataset_standard
        if pass_schema_hint_to_fact_extractor and dataset_standard is not None
        else None
    )
    facts_result = extract_facts_from_text(
        text=text,
        max_facts=max_facts,
        model_name=model_name,
        temperature=temperature,
        provider=provider,
        schema_standard=schema_hint,
    )

    schema_output = None
    if use_llm_dataset_builder:
        if dataset_standard is None:
            raise ValueError("dataset_standard must be provided when use_llm_dataset_builder=True")

        schema_output = llm_build_dataset_from_facts(
            facts_or_result=facts_result,
            dataset_standard=dataset_standard,
            raw_text=text,
            records_key=dataset_records_key,
            record_class_name=record_class_name,
            output_class_name=output_class_name,
            document_excerpt_max_chars=document_excerpt_max_chars,
            model_name=model_name,
            temperature=temperature,
            provider=provider,
        )

        df = build_dataset_from_schema_output(schema_output, records_key=dataset_records_key)
    else:
        df = build_fact_dataset(facts_result)

    if use_llm_dataset_builder:
        ok, issues = validate_schema_records_dataset(df)
    else:
        ok, issues = validate_fact_dataset(df)

    out: Dict[str, Any] = {
        "workflow": workflow,
        "facts_result": facts_result,
        "dataset": df,
        "validation_ok": ok,
        "validation_issues": issues,
    }
    if schema_output is not None:
        out["schema_output"] = schema_output

    logger.info(
        "run_two_step_text_to_dataset: completed (dataset_rows=%d, validation_ok=%s)",
        len(df),
        ok,
    )
    return out

