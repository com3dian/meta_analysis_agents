"""
Two-step static workflow: raw text → facts → structured dataset.

This mirrors a simple two-agent pipeline:

1) Fact Extractor
   - Input: raw document text
   - Output: structured list of (subject, predicate, object, source_span)

2) Dataset Builder / Validator
   - Input: list of facts
   - Output: pandas DataFrame + basic validation report
"""

import json
import logging
import re
from typing import List, Tuple, Optional, Dict, Any, Union, Type

import pandas as pd
from pydantic import BaseModel, Field

from src.direct_llm_call.utils import invoke_llm_with_structured_output
from src.core.schema_factory import create_output_schema

logger = logging.getLogger(__name__)


class Fact(BaseModel):
    """
    Single extracted fact in (subject, predicate, object) form.
    """

    subject: str = Field(
        ...,
        description="The main entity the fact is about.",
        examples=["Cassava", "Nitrogen fertilizer", "Intercropping system"],
    )
    predicate: str = Field(
        ...,
        description="The relationship or property being asserted.",
        examples=["increases yield", "reduces weed biomass"],
    )
    object: str = Field(
        ...,
        description="The value, quantity, or secondary entity.",
        examples=["by 20%", "soil nitrogen content", "cowpea"],
    )
    source_span: Optional[str] = Field(
        None,
        description="Short quote or sentence from which this fact was derived.",
    )


class FactExtractionResult(BaseModel):
    """
    Wrapper model for structured LLM output.
    """

    facts: List[Fact] = Field(
        default_factory=list,
        description="List of extracted facts from the input document.",
    )


FACT_PROMPT_TEMPLATE = """You are an information extraction assistant.

Your task is to read the following document text and extract concise, atomic facts
as triples (subject, predicate, object). Each fact should be:
- As specific as possible
- Grounded in the text
- Useful for quantitative or qualitative analysis later

Document:
\"\"\"{text}\"\"\"

Instructions:
- Return at most {max_facts} of the most important and distinct facts.
- For each fact, fill in:
  - subject: the main entity (e.g., a crop, treatment, variable)
  - predicate: the relationship or property (e.g., 'increases', 'reduces', 'is associated with')
  - object: the value, quantity, or target entity
  - source_span: a short quote or sentence from the document that supports the fact

Return ONLY structured data according to the provided schema.
"""


def extract_facts_from_text(
    text: str,
    max_facts: int = 20,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    provider: Optional[str] = None,
) -> FactExtractionResult:
    """
    Step 1: Fact Extractor (Text → Facts).

    Uses a structured-output LLM call to convert raw text into a list of `Fact`.
    This is analogous to a first agent whose sole responsibility is extraction.

    Args:
        text: Raw input document text.
        max_facts: Maximum number of facts to extract.
        model_name: Optional LLM model override.
        temperature: LLM temperature (0.0 for deterministic output).
        provider: Optional provider override (google, openai, surf, qwen).

    Returns:
        FactExtractionResult containing a list of facts.
    """
    logger.info(
        "extract_facts_from_text: starting (text_len=%d, max_facts=%d)",
        len(text),
        max_facts,
    )
    prompt = FACT_PROMPT_TEMPLATE.format(text=text, max_facts=max_facts)
    logger.debug("extract_facts_from_text: prompt_len=%d", len(prompt))

    result = invoke_llm_with_structured_output(
        prompt=prompt,
        output_schema=FactExtractionResult,
        model_name=model_name,
        temperature=temperature,
        provider=provider,
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


DATASET_BUILDER_PROMPT_TEMPLATE = """You are a dataset builder.

You will be given a list of extracted facts (subject, predicate, object, source_span).
Your job is to convert these facts into a structured dataset that follows the schema below.
{document_section}

**Target schema (field names and descriptions):**
{schema_descriptions}

**Extracted facts (JSON):**
{facts_json}

**Rules:**
- Output must strictly follow the schema. Each record must have exactly the fields listed above.
- Map fact triples (subject, predicate, object) to the schema fields using the descriptions as guidance.
- Prefer copying exact strings from facts to reduce hallucination.
- Create as many records as needed (including 0 if nothing applies).
- Do not invent values not present in the facts or document.
{fill_rules}

Return ONLY structured data according to the provided schema.
"""

DATASET_BUILDER_FILL_RULES_FACTS_ONLY = """
- If a field cannot be inferred from the facts, set it to null.
"""

DATASET_BUILDER_FILL_RULES_WITH_DOCUMENT = """
- When a field cannot be inferred from the facts alone, search the document excerpt above for that information (e.g., Materials and Methods for treatment/tillage, Abstract/Site description for soil/location).
- If you find the value in the document, use it. If the document explicitly indicates a field was considered but not reported, use "Not explicitly stated".
- If the field is truly absent from both facts and document, set it to null.
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


def _truncate_document_for_prompt(text: str, max_chars: int = 15000) -> str:
    """Truncate document to stay within prompt limits while preserving key sections."""
    if len(text) <= max_chars:
        return text
    # Keep start (often Abstract, Intro) and end (often Methods, Results)
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars - 100  # 100 for ellipsis
    return (
        text[:head_chars]
        + "\n\n[... document truncated ...]\n\n"
        + text[-tail_chars:]
    )


def llm_build_dataset_from_facts(
    facts_or_result: Union[FactExtractionResult, List[Fact]],
    dataset_standard: Union[str, Dict[str, Any]],
    *,
    raw_text: Optional[str] = None,
    records_key: str = "records",
    record_class_name: str = "Record",
    output_class_name: str = "DatasetOutput",
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
    logger.debug("llm_build_dataset_from_facts: facts_json_len=%d", len(facts_json))

    if raw_text:
        doc_excerpt = _truncate_document_for_prompt(raw_text)
        document_section = f"""
**Source document (excerpt; use to fill fields not present in facts):**
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
    )

    result = invoke_llm_with_structured_output(
        prompt=prompt,
        output_schema=OutputSchema,
        model_name=model_name,
        temperature=temperature,
        provider=provider,
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
    max_facts: int = 20,
    dataset_standard: Optional[Union[str, Dict[str, Any]]] = None,
    dataset_records_key: str = "records",
    use_llm_dataset_builder: bool = False,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    End-to-end helper: Raw text → Facts → Validated Dataset.

    This function is the static-workflow equivalent of a small multi-agent
    system: internally it runs the Fact Extractor and Dataset Builder /
    Validator in sequence, but presents a single high-level API.

    Args:
        text: Raw input document text.
        max_facts: Maximum number of facts to extract.
        model_name: Optional LLM model override.
        temperature: LLM temperature for extraction.
        provider: Optional provider override.

    Returns:
        Dictionary with:
        - 'facts_result': FactExtractionResult
        - 'dataset': pandas DataFrame (facts-as-rows by default, or schema-built if enabled)
        - 'validation_ok': bool
        - 'validation_issues': list of strings
        - 'schema_output' (optional): structured output from dataset-builder LLM
    """
    logger.info(
        "run_two_step_text_to_dataset: starting (text_len=%d, max_facts=%d, use_llm_dataset_builder=%s)",
        len(text),
        max_facts,
        use_llm_dataset_builder,
    )

    # Step 1: Text → Facts
    facts_result = extract_facts_from_text(
        text=text,
        max_facts=max_facts,
        model_name=model_name,
        temperature=temperature,
        provider=provider,
    )

    schema_output = None
    if use_llm_dataset_builder:
        if dataset_standard is None:
            raise ValueError("dataset_standard must be provided when use_llm_dataset_builder=True")

        # Step 2 (LLM): Facts → Schema-conforming dataset (structured output)
        # Pass raw_text so the dataset builder can fill fields not present in facts
        schema_output = llm_build_dataset_from_facts(
            facts_or_result=facts_result,
            dataset_standard=dataset_standard,
            raw_text=text,
            records_key=dataset_records_key,
            model_name=model_name,
            temperature=temperature,
            provider=provider,
        )

        # Convert schema output to DataFrame
        df = build_dataset_from_schema_output(schema_output, records_key=dataset_records_key)
    else:
        # Step 2a (deterministic): Facts → DataFrame
        df = build_fact_dataset(facts_result)

    # Step 2b: Validate DataFrame
    ok, issues = validate_fact_dataset(df)

    out: Dict[str, Any] = {
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


