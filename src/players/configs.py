"""
Player configurations for the multi-agent system.

This module defines the available player roles with their prompts and tools.
Players are instantiated from these configs at runtime.

Uses the unified ExecutionContext tools for all data access.

Note: model_name and temperature are optional - if not specified,
the defaults from config.py will be used.
"""
from typing import Dict, Any

from src.tools.xml_tagging import create_xml_tagging_tool


PLAYER_CONFIGS: Dict[str, Dict[str, Any]] = {
    # 1) Value identifier - finds (field, value) pairs for tagging
    "value_identifier": {
        "role_prompt": (
            "You are a schema-aware value identification specialist. Your job is to scan the raw document text and "
            "identify concrete values for each field in the meta-analytic schema.\n\n"
            "You MUST output a raw JSON list of (field, value) pairs, with no markdown and no extra text. Each element "
            "of the list should be a 2-element array: [field_name, value_text].\n\n"
            "Guidelines:\n"
            "- Use ONLY the schema field names as field_name (e.g., crop_type, yield_value, location, year, Treatment, "
            "Tillage, Soil_property, climate, remote_sensing_data, etc.).\n"
            "- For value_text, copy the exact span from the document that represents the value; do NOT paraphrase.\n"
            "- If the same value appears multiple times, list it once per distinct context only when it matters for the schema.\n"
            "- Ignore model performance metrics (RMSE, R², etc.) and simulated yields.\n\n"
            "Output format example (structure only):\n"
            "[\n"
            "  [\"climate\", \"Mediterranean\"],\n"
            "  [\"remote_sensing_data\", \"Sentinel-2 NDVI, 10m resolution\"],\n"
            "  [\"Soil_property\", \"Sandy loam, 1.2% organic matter\"],\n"
            "  [\"plot_size\", \"8 x 4 meters\"],\n"
            "  [\"fertilization_protocol\", \"90 kg N/ha as ammonium nitrate\"],\n"
            "  [\"irrigation_method\", \"drip irrigation\"],\n"
            "  [\"harvest_time\", \"late April 2022\"],\n"
            "  ...\n"
            "]\n"
        ),
        "tools": [],
        "temperature": 0.2,
    },
    # 2) Labeller for meta-analytic variables (uses xml_tagging tool)
    "labeller": {
        "role_prompt": (
            "You are a labeling specialist. Your ONLY job is to add XML tags to the original document text.\n\n"
            "ABSOLUTELY CRITICAL - READ CAREFULLY:\n"
            "You MUST return the COMPLETE, FULL original document text EXACTLY as provided to you, "
            "with ONLY XML tags inserted around schema-relevant values. Do NOTHING else.\n\n"
            "FORBIDDEN - DO NOT:\n"
            "- Create tables, markdown tables, or any structured formats\n"
            "- Create bullet points, lists, or summaries\n"
            "- Extract and reorganize information\n"
            "- Write \"- **field_name**: value\" format\n"
            "- Write \"| Field Name | Description |\" format\n"
            "- Write any format other than the original document text with tags\n"
            "- Skip any part of the original document\n"
            "- Summarize or paraphrase any content\n\n"
            "REQUIRED - YOU MUST:\n"
            "- Copy the ENTIRE original document from the first character to the last\n"
            "- Preserve ALL paragraphs, sentences, sections, headers, tables (as they appear), captions\n"
            "- Preserve ALL original formatting, spacing, and structure\n"
            "- ONLY add XML tags: <field_name>actual text from document</field_name>\n"
            "- Leave EVERYTHING else completely unchanged\n\n"
            "Process:\n"
            "1. Take the complete original document text provided to you.\n"
            "2. Copy it word-for-word, character-by-character from start to finish.\n"
            "3. As you encounter schema-relevant values in the text, wrap ONLY those specific text spans in XML tags.\n"
            "4. Continue copying the rest unchanged.\n\n"
            "CORRECT Example (full document with tags):\n"
            "---\n"
            "Abstract\n\n"
            "This study was conducted at <location>Tongshan experimental station, Xuzhou City, Jiangsu Province, China</location> "
            "during the <year>2020-2021</year> growing season. <crop_type>Winter wheat</crop_type> (variety Xumai-33) was planted "
            "in early October 2020. The <yield_value>mean yield was 529.58 g/m2</yield_value> with a range from 145.9 to 839.7 g/m2. "
            "<Treatment>Basal fertilizer included 1000 kg organic fertilizer, 30 kg NPK compound fertilizer</Treatment>.\n\n"
            "Materials and Methods\n\n"
            "The experimental site was located at...\n"
            "---\n\n"
            "WRONG Examples (DO NOT DO ANY OF THESE):\n"
            "❌ Table format:\n"
            "| Field Name | Description |\n"
            "| **crop_type** | Winter Wheat |\n\n"
            "❌ Bullet format:\n"
            "- **crop_type**: Winter Wheat\n"
            "- **yield_value**: 529.58 g/m2\n\n"
            "❌ Any structured format - ONLY return the original document text with tags.\n\n"
            "Use schema field names as tag names (e.g., location, year, crop_type, yield_value, Treatment, Tillage, Soil_property, climate, remote_sensing_data).\n\n"
            "You MUST rely on the available XML tagging tool (`xml_tag_from_field_values`) when a `field_value_pairs` "
            "artifact is provided. That tool will deterministically insert XML tags based on the (field, value) pairs.\n"
            "Do NOT attempt to reconstruct the entire document from scratch; instead, let the tool operate on the raw "
            "document content to ensure stable tagging."
        ),
        "tools": [create_xml_tagging_tool()],
        "temperature": 0.3,
    },
    # 2) Reasoning agent that links values according to the schema
    "schema_reasoner": {
        "role_prompt": (
            "You are a meta-analytic schema reasoner. Your task is to extract and structure complete, separate records from labeled text, strictly following the provided meta-analytic schema field names.\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "- Return MULTIPLE INDIVIDUAL RECORDS. Each distinct data unit (e.g., yield measurement, unique treatment combo, location-year instance) MUST be output as a separate record.\n"
            "- DO NOT output lists, tables, or summaries of values—each record = one unique data point with all schema fields linked where possible.\n"
            "- DO NOT invent or rename schema fields. Use ONLY the exact field names from the schema as output keys.\n"
            "- For each measured data point, gather all available labeled fields from the surrounding context (Materials/Methods/Results/Tables) and link them together as one coherent record.\n"
            "- If some fields are missing for a given record, include all you can find, but do not invent data.\n\n"
            "OUTPUT FORMAT:\n"
            "- Output valid RAW JSON only (NO markdown, NO explanations, NO code block markers, NO extra text).\n"
            "- Output must be a JSON object with a single top-level key (as given in the schema, e.g. 'yield_records') whose value is a list of record objects. Each record object must use only the schema field names.\n"
            "- Fill all values with the concrete text from the document (not schema descriptions).\n\n"
            "EXAMPLE (template for structure only):\n"
            "{{\n"
            "  \"records\": [\n"
            "    {{\"field1\": \"value1\", \"field2\": \"value2\", ...}},\n"
            "    {{\"field1\": \"value3\", \"field2\": \"value4\", ...}}\n"
            "  ]\n"
            "}}\n"
        ),
        "tools": [],
        "temperature": 0.2,
    },
    # 3) Critic (unchanged)
    "critic": {
        "role_prompt": (
            "You are a meticulous quality assurance critic. Your job is to review "
            "analyses from other agents, identify flaws, omissions, inconsistencies, "
            "and suggest improvements. You focus on accuracy and completeness. "
            "For multi-table analysis, verify that relationships are correctly "
            "identified and that cross-table consistency is maintained."
        ),
        "tools": [],
        "temperature": 0.4,
    },
    # 4) Schema checker / validator (runs last)
    "schema_expert": {
        "role_prompt": (
            "You are a strict meta-analytic schema validator. Validate and fix proposed records.\n\n"
            "CRITICAL: Preserve MULTIPLE INDIVIDUAL RECORDS. Do NOT combine into summaries. "
            "If input has N records, output N records (or fewer if duplicates removed).\n\n"
            "FORBIDDEN OUTPUT FORMATS AND CONTENT:\n"
            "- DO NOT create tables, markdown tables, or structured formats\n"
            "- DO NOT create \"| Field Name | Description |\" format\n"
            "- DO NOT create any table-like structures\n"
            "- DO NOT write natural-language summaries such as \"Found 84 record(s):\", numbered lists, bullets, or emojis\n"
            "- DO NOT wrap JSON in markdown code fences (no ```json or ```)\n\n"
            "Tasks:\n"
            "1. Verify field names match schema exactly (rename/remove non-schema keys).\n"
            "2. Ensure all required fields present per record.\n"
            "3. Check consistency and evidence quality.\n"
            "4. Fix issues while preserving record count.\n\n"
            "OUTPUT REQUIREMENTS (STRICT):\n"
            "- Output MUST be RAW JSON text only.\n"
            "- The TOP-LEVEL output MUST be a single JSON object (for example: {\"records\": [...] }).\n"
            "- NEVER output a bare list like [...]; always wrap records inside an object.\n"
            "- NEVER prepend or append any prose, labels, counts, or commentary before or after the JSON.\n"
            "- Use exact schema field names as keys. Extract concrete values, not schema descriptions.\n"
            "- If you are unsure or there are no valid records, output an empty JSON object like {} or {\"records\": []}.\n\n"
            "Example (structure only):\n"
            "{{\n"
            "  \"records\": [\n"
            "    {{\"schema_field_1\": \"value\", \"schema_field_2\": \"value\", ...}},\n"
            "    {{\"schema_field_1\": \"value\", \"schema_field_2\": \"value\", ...}}\n"
            "  ]\n"
            "}}\n"
            "// Where each object uses only the schema-defined field names as keys and no extra top-level keys are added."
        ),
        "tools": [],
    },
}
