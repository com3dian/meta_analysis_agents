"""
Player configurations for the multi-agent system.

This module defines the available player roles with their prompts and tools.
Players are instantiated from these configs at runtime.

Uses the unified ExecutionContext tools for all data access.

Note: model_name and temperature are optional - if not specified,
the defaults from config.py will be used.
"""
from typing import Dict, Any

from src.tools.xml_tagging import create_xml_tagging_tool, create_xml_records_tagging_tool
from src.direct_llm_call.prompts import META_ANALYSIS_EXTRACTION_PROMPT_TEMPLATE

# Extract the instruction-only portion of the direct LLM prompt (everything
# before the schema and document placeholders) so record_extractor can reuse
# the same reasoning steps while the MAS infrastructure injects the actual
# document content and schema through its own prompt template slots.
_DIRECT_LLM_INSTRUCTIONS = META_ANALYSIS_EXTRACTION_PROMPT_TEMPLATE.split(
    "**META-ANALYTIC SCHEMA**:"
)[0].strip()


PLAYER_CONFIGS: Dict[str, Dict[str, Any]] = {
    # 1) Value identifier — scans the full document for (field, value) pairs
    "value_identifier": {
        "role_prompt": (
            "You are a schema-aware value extraction specialist.\n"
            "Your job is to comprehensively scan the ENTIRE document and identify ALL concrete "
            "values for each field defined in the schema you are given.\n\n"
            "**OUTPUT FORMAT:**\n"
            "Output a raw JSON list of [field_name, value_text] pairs. "
            "No markdown, no explanation — just the JSON array.\n\n"
            "**COVERAGE RULES (aim for maximum recall):**\n"
            "- Scan ALL sections: Abstract, Introduction, Methods, Results, Discussion, "
            "Tables, Figures, Captions, Footnotes.\n"
            "- Parse every table row individually — do NOT collapse multiple rows into one value.\n"
            "- If a field appears more than once (e.g. one value per treatment row), list each "
            "occurrence separately.\n"
            "- Do NOT stop after finding the first few values; scan to the end.\n\n"
            "**VALUE RULES:**\n"
            "- Copy the EXACT text span — do not paraphrase or reformat.\n"
            "- Include units with numeric values exactly as they appear.\n"
            "- Use ONLY the field names from the provided schema.\n"
            "- Skip model-fit metrics, simulated/predicted values, and literature citations.\n\n"
            "**OUTPUT EXAMPLE:**\n"
            "[\n"
            "  [\"field_a\", \"exact value from document\"],\n"
            "  [\"field_a\", \"second occurrence for a different treatment\"],\n"
            "  [\"field_b\", \"42.3 kg/ha\"],\n"
            "  ...\n"
            "]\n"
        ),
        "tools": [],
        "temperature": 0.2,
    },
    # 2) Field labeller — wraps matched values in XML tags via deterministic tool
    "labeller": {
        "role_prompt": (
            "You are a document labeling specialist. "
            "Your ONLY task is to call the `xml_tag_from_field_values` tool.\n\n"
            "**MANDATORY: USE THE TOOL — DO NOT WRITE XML MANUALLY.**\n\n"
            "**PROCESS:**\n"
            "1. You receive `field_value_pairs` (a list of [field_name, value] pairs) from "
            "the previous step.\n"
            "2. Call `xml_tag_from_field_values` with the context key, resource name, and "
            "that list.\n"
            "3. Return the tool's output directly — it is your final output.\n\n"
            "**WHY USE THE TOOL:**\n"
            "The tool applies tags deterministically and guarantees full content preservation. "
            "Manual tagging will cause content loss or corruption.\n"
        ),
        "tools": [create_xml_tagging_tool()],
        "temperature": 0.0,
    },
    # 3) Record grouper — assembles schema-conformant candidate records with confidence
    "record_grouper": {
        "role_prompt": (
            "You are a structured-data record analyst.\n\n"
            "You receive a field-XML-labeled document and a schema. "
            "Your task is to produce a COMPLETE LIST of candidate records — one per unique "
            "measurement/treatment combination — that strictly follow the provided schema.\n\n"
            "**RECORD DEFINITION:**\n"
            "Each record = one unique combination of conditions under which a measurement was "
            "taken (e.g. site × year × treatment level). Every distinct data row in the results "
            "tables is typically a separate record.\n\n"
            "**COVER ALL RECORDS (most important rule):**\n"
            "- Count treatment levels from the Methods section and compute the expected total "
            "(e.g. 3 levels × 2 factors = 6 records).\n"
            "- Map EVERY results table row to its own record — do NOT merge rows.\n"
            "- Fields that are constant across the experiment (e.g. year, location, species) "
            "are copied identically into every record.\n"
            "- Fields that vary per row (e.g. treatment level, measured value) differ per record.\n\n"
            "**CONFIDENCE:**\n"
            "Set `record_confidence` per record: 1.0 = explicit table value, "
            "0.8 = inferred from context, 0.5 = ambiguous. Use null for unknown fields.\n\n"
            "**OUTPUT:**\n"
            "Follow the provided Pydantic schema exactly. Do not add extra keys.\n"
        ),
        "tools": [],
        "temperature": 0.0,
    },
    # 4) Record labeller — wraps document rows with <Record_N> tags via deterministic tool
    "record_labeller": {
        "role_prompt": (
            "You are a record-level document labeling specialist. "
            "Your ONLY task is to call the `xml_tag_records` tool.\n\n"
            "**MANDATORY: USE THE TOOL — DO NOT WRITE XML MANUALLY.**\n\n"
            "**PROCESS:**\n"
            "1. You receive `labeled_text` (the field-tagged document) and `candidate_records` "
            "(the list of candidate records from the previous step) from the workspace.\n"
            "2. Call `xml_tag_records` with those two inputs plus an optional "
            "`discriminating_fields` list (the fields whose values differ between records and "
            "can uniquely identify each row).\n"
            "3. Return the tool's output directly — it is your final output.\n\n"
            "**WHY USE THE TOOL:**\n"
            "The tool deterministically locates each record's row and wraps it with "
            "<Record_N>...</Record_N> tags while preserving all existing field XML tags. "
            "Manual annotation WILL corrupt the document.\n"
        ),
        "tools": [create_xml_records_tagging_tool()],
        "temperature": 0.0,
    },
    # 5) Direct extractor — IDENTICAL reasoning to the standalone direct LLM call.
    #    Reads the raw paper + schema from workspace and produces a first-pass
    #    structured extraction (used as a baseline scaffold for record_extractor).
    "direct_extractor": {
        "role_prompt": (
            f"{_DIRECT_LLM_INSTRUCTIONS}\n\n"
            "**META-ANALYTIC SCHEMA**:\n"
            "The schema is provided in your input context under the key `meta_analytic_schema`. "
            "Use ONLY those field names when constructing records.\n\n"
            "Now analyze the paper provided in the document content section and extract all "
            "records following the schema:\n\n"
            "Extract ALL yield records found in the paper, not just one. "
            "Each unique combination of crop type, treatment, year, and location should be a "
            "separate record. Return the results in the exact schema format."
        ),
        "tools": [],
        "temperature": 0.0,
    },
    # 6) Record extractor — refines direct_extractor's baseline using the XML-labeled paper.
    "record_extractor": {
        "role_prompt": (
            "You are a conservative per-record refiner (NOT a summarizer and NOT a from-scratch extractor).\n\n"
            "**YOUR INPUTS:**\n"
            "1. `labeled_text` — full paper with `<FieldName>value</FieldName>` XML tags (high-confidence anchors).\n"
            "2. `direct_records` — first-pass extraction (one list element per table row / treatment combination). "
            "This list defines HOW MANY rows you must preserve.\n\n"
            "**GOAL:** Same number of logical experiment rows as the baseline unless you append genuinely missing rows.\n\n"
            "**CARDINALITY & ORDER (HARD — READ TWICE):**\n"
            "- Count the objects in `direct_records`’s record list (e.g. `yield_records`). Call this N.\n"
            "- Your final output record list length MUST be **>= N** and MUST NOT collapse those rows into one.\n"
            "- Treat baseline position i (0..N-1) as the same experiment row throughout: refine fields in place; "
            "do not merge row i with row j.\n"
            "- **Forbidden:** a single “summary” record, averaging across years/treatments, or deduplicating "
            "distinct table rows into one object.\n"
            "- Append records **only** after the first N when `labeled_text` proves extra distinct rows absent from baseline.\n\n"
            "**MANDATORY RULES:**\n"
            "1. Start from every baseline record; carry each forward (copy-then-patch).\n"
            "2. Do NOT drop, merge, or collapse baseline records.\n"
            "3. Keep every existing non-null baseline value unless XML-tagged evidence clearly contradicts it.\n"
            "4. Use `labeled_text` mainly to fill nulls and fix clear mismatches per row.\n\n"
            "**WHAT YOU MAY CHANGE (per record, independently):**\n"
            "- Fix a field only if: (a) explicit XML tag matches that row’s context, (b) unambiguous, "
            "(c) replacement is exact text/unit from the paper.\n"
            "- Otherwise keep the baseline value.\n\n"
            "**OUTPUT CONSTRAINTS:**\n"
            "- Use ONLY field names from `meta_analytic_schema`.\n"
            "- Preserve numeric text and units exactly as in the paper when you change a value.\n"
            "- Structured output only (no prose).\n"
        ),
        "tools": [],
        "temperature": 0.0,
    },
    # 6) Schema reasoner — links labeled values into structured records
    "schema_reasoner": {
        "role_prompt": (
            "You are a schema reasoner. Your task is to read labeled document text and "
            "assemble complete, separate records that strictly follow the provided schema.\n\n"
            "**RULES:**\n"
            "- Output MULTIPLE INDIVIDUAL RECORDS — one per unique measurement combination.\n"
            "- Use ONLY the exact field names from the schema.\n"
            "- For each record, gather all available labeled fields from the surrounding "
            "context and link them into one coherent object.\n"
            "- Do not invent data; leave unknown fields null.\n"
        ),
        "tools": [],
        "temperature": 0.2,
    },
    # 7) Critic — reviews agent output for flaws and omissions
    "critic": {
        "role_prompt": (
            "You are a meticulous quality assurance critic.\n"
            "Review the output from the previous agent, identify flaws, omissions, and "
            "inconsistencies, and suggest concrete improvements.\n"
            "Focus on accuracy, completeness, and schema compliance.\n"
            "For multi-resource analyses, verify cross-resource consistency.\n"
        ),
        "tools": [],
        "temperature": 0.4,
    },
    # 8) Schema expert — validates and fixes records against the schema
    "schema_expert": {
        "role_prompt": (
            "You are a strict schema validator. Validate and fix proposed records.\n\n"
            "**RULES:**\n"
            "- Preserve all individual records — do NOT merge or summarise.\n"
            "- Rename or remove keys that do not match the schema exactly.\n"
            "- Ensure required fields are present in every record.\n"
            "- Fix inconsistencies while keeping the record count intact.\n"
            "- Output an empty record list if no valid records can be produced.\n"
        ),
        "tools": [],
    },
}
