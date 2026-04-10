"""
This file stores all prompt templates for the multi-agent system.
"""
from langchain_core.prompts import ChatPromptTemplate


def get_planning_prompt() -> ChatPromptTemplate:
    """
    Returns the ``ChatPromptTemplate`` used by the planning orchestrator.

    Template variables
    ------------------
    - **objective** : high-level goal the planner should achieve.
    - **available_players** : human-readable description of all available players.
    - **format_instructions** : JSON schema the planner must output.
    - **context_info** : brief description of the execution context.
    """
    system_prompt = """\
You are an expert planner agent for a multi-step document extraction workflow.
Given an overall objective, brief context information, and a list of available players,
produce a clear, dataflow-consistent step-by-step plan.

━━━ REQUIRED PLAN STRUCTURE (follow exactly) ━━━

Produce exactly 4 steps in this order:

  Step 1 — value_identifier
    Task   : Scan the FULL document and extract every (field, value) pair for every
             field in the schema.  Cover all sections, table rows, figure captions,
             and footnotes.  List each occurrence separately — do NOT collapse rows.
             Use ONLY the field names defined in meta_analytic_schema.
    Inputs : {{"meta_analytic_schema": "meta_analytic_schema"}}
    Outputs: [field_value_pairs]

  Step 2 — labeller
    Task   : Use the xml_tag_from_field_values tool to wrap each matched value in the
             document with XML field tags.  Do not modify any original text.
    Inputs : {{"field_value_pairs": "field_value_pairs"}}
    Outputs: [labeled_text]

  Step 3 — direct_extractor
    Task   : Apply the identical multi-step extraction reasoning as the standalone
             direct LLM call: anchor identification, contextual reasoning, completeness
             & confidence, record construction.  Read the raw paper content directly
             from context (NOT the labeled text) and produce a first-pass extraction
             of ALL yield records.  Use the schema from meta_analytic_schema — exact
             field names, null for missing values.
    Inputs : {{"meta_analytic_schema": "meta_analytic_schema"}}
    Outputs: [direct_records]

  Step 4 — record_extractor
    Task   : Refine direct_records WITHOUT reducing row count: keep the same
             number of baseline records in the same order (one output row per
             baseline row); only fill nulls or fix values when labeled_text
             clearly supports it per row. Append extra records ONLY if labeled_text
             proves distinct table rows missing from direct_records. Never merge
             baseline rows into a single summary record.
    Inputs : {{"labeled_text": "labeled_text", "direct_records": "direct_records", "meta_analytic_schema": "meta_analytic_schema"}}
    Outputs: [final_meta_analysis_records]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Schema handling (CRITICAL):
- Treat JSON keys in the schema as the EXACT field names for every record.
- Schema descriptions are guidance only — record values must be concrete text from the document.
- Do not rename, add, or remove schema fields in any output artifact.
- final_meta_analysis_records objects must use only schema-defined field names.

Dataflow rules (CRITICAL):
- Every artifact named in a step's inputs MUST have been produced by an earlier step's outputs,
  OR be one of the pre-populated workspace artifacts: meta_analytic_schema.
- Never use the ExecutionContext name or any resource name as an artifact name.

Player assignment — use exact names from the available players list:
- value_identifier : extracts (field, value) pairs from the full document
- labeller         : applies XML field tags via the xml_tag_from_field_values tool
- direct_extractor : first-pass extraction using identical reasoning to the direct LLM call
- record_extractor : refines direct_extractor's baseline using XML-labeled evidence

Artifact names to use consistently:
- meta_analytic_schema        (pre-populated — the JSON schema, available from Step 1 onward)
- field_value_pairs           (Step 1 output)
- labeled_text                (Step 2 output)
- direct_records              (Step 3 output)
- final_meta_analysis_records (Step 4 output)

Context:
{context_info}

Available players:
{available_players}

Output:
Return ONLY a JSON object that strictly follows this schema:
{format_instructions}
"""

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Generate a complete plan to achieve the following objective: '{objective}'.",
            ),
        ]
    )


def get_short_planning_prompt() -> ChatPromptTemplate:
    """
    Returns the ``ChatPromptTemplate`` for the compact 3-step pipeline:

      Step 1 — value_identifier  →  field_value_pairs
      Step 2 — labeller          →  labeled_text
      Step 3 — record_extractor  →  final_meta_analysis_records

    This skips the record_grouper / record_labeller steps so the final
    extraction step behaves close to the direct LLM call but benefits from
    pre-labeled field tags in the document.
    """
    system_prompt = """\
You are an expert planner agent for a multi-step document extraction workflow.
Given an overall objective, brief context information, and a list of available players,
produce a clear, dataflow-consistent step-by-step plan.

━━━ REQUIRED PLAN STRUCTURE (follow exactly) ━━━

Produce exactly 3 steps in this order:

  Step 1 — value_identifier
    Task   : Scan the FULL document and extract every (field, value) pair for every
             field in the schema.  Cover all sections, table rows, figure captions,
             and footnotes.  List each occurrence separately — do NOT collapse rows.
             Use ONLY the field names defined in meta_analytic_schema.
    Inputs : {{"meta_analytic_schema": "meta_analytic_schema"}}
    Outputs: [field_value_pairs]

  Step 2 — labeller
    Task   : Use the xml_tag_from_field_values tool to wrap each matched value in the
             document with XML field tags.  Do not modify any original text.
    Inputs : {{"field_value_pairs": "field_value_pairs"}}
    Outputs: [labeled_text]

  Step 3 — record_extractor
    Task   : Read the field-tagged document and extract ALL schema-conformant records.
             Apply the full multi-step reasoning process:
             (1) locate every yield anchor in the labeled document,
             (2) gather surrounding context fields for each anchor,
             (3) fill missing fields from Methods/Site/Climate sections,
             (4) construct one record per unique treatment/crop combination.
             Each distinct data row in the results tables is a separate record.
             Do NOT merge rows. Use the XML field tags as high-confidence anchors.
    Inputs : {{"labeled_text": "labeled_text"}}
    Outputs: [final_meta_analysis_records]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Schema handling (CRITICAL):
- Treat JSON keys in the schema as the EXACT field names for every record.
- Schema descriptions are guidance only — record values must be concrete text from the document.
- Do not rename, add, or remove schema fields in any output artifact.

Dataflow rules (CRITICAL):
- Every artifact named in a step's inputs MUST have been produced by an earlier step's outputs,
  OR be one of the pre-populated workspace artifacts: meta_analytic_schema.
- Never use the ExecutionContext name or any resource name as an artifact name.

Player assignment — use exact names from the available players list:
- value_identifier : extracts (field, value) pairs from the full document
- labeller         : applies XML field tags via the xml_tag_from_field_values tool
- record_extractor : applies full extraction reasoning and produces final structured records

Artifact names to use consistently:
- meta_analytic_schema        (pre-populated — the JSON schema, available from Step 1 onward)
- field_value_pairs           (Step 1 output)
- labeled_text                (Step 2 output)
- final_meta_analysis_records (Step 3 output)

Context:
{context_info}

Available players:
{available_players}

Output:
Return ONLY a JSON object that strictly follows this schema:
{format_instructions}
"""

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Generate a complete plan to achieve the following objective: '{objective}'.",
            ),
        ]
    )


def get_task_execution_prompt() -> ChatPromptTemplate:
    """
    Returns the prompt template for task execution by a player.
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are {player_name}. {role_prompt}

You are executing a specific task as part of a larger workflow.
Your goal is to complete the task thoroughly and provide actionable results.

**Available Tools:**
{tool_descriptions}
""",
            ),
            (
                "human",
                """**Task:** {task}

**Context Information:**
{context_info}

**Target Resources for This Step:** {target_resources}

**Context from Previous Steps:**
{input_context}

**Tool Results:**
{tool_results}

Execute this task and provide a comprehensive response.
""",
            ),
        ]
    )


def get_initial_work_prompt() -> ChatPromptTemplate:
    """
    Returns the prompt for generating initial work in a debate.
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are {player_name}. {role_prompt}

You are participating in a multi-agent debate. Your goal is to provide
your unique perspective and insights based on your expertise.
""",
            ),
            (
                "human",
                """**Task:** {task}

**Context Information:**
{context_info}

**Target Resources:** {target_resources}

**Available Context:**
{context}

Provide your initial analysis.
""",
            ),
        ]
    )


def get_critique_prompt() -> ChatPromptTemplate:
    """
    Returns the prompt for critiquing other players' work.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are {player_name}. {role_prompt}"),
            (
                "human",
                """**Task being analyzed:** {task}

**Work from other players to critique:**

{other_work}

Provide your detailed critique.
""",
            ),
        ]
    )


def get_revision_prompt() -> ChatPromptTemplate:
    """
    Returns the prompt for revising work based on critiques.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are {player_name}. {role_prompt}"),
            (
                "human",
                """**Task:** {task}

**Your Original Analysis:**
{original_work}

**Critiques Received:**
{critiques}

Provide your revised analysis.
""",
            ),
        ]
    )


def get_synthesis_prompt() -> ChatPromptTemplate:
    """
    Returns the prompt for synthesizing multiple analyses into one.
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a synthesis expert responsible for combining multiple analyses into a single, structured output.
- Be CONCISE: Output only the essential information.
- Be STRUCTURED: Use a clean key-value format or JSON structure.
- NO lengthy explanations or narratives.
- Focus on FACTS, not process descriptions.

**CRITICAL SCHEMA COMPLIANCE**:
- If a meta-analytic schema is provided in the analyses or workspace, you MUST use EXACTLY the field names from that schema.
- Do NOT invent new field names or rename schema fields.
- Extract values from the analyses and map them to the exact schema field names (e.g., 'crop_type', 'yield_value', 'location', 'year', 'Treatment', 'Tillage', 'Soil_property', 'climate', 'remote_sensing_data').
- The schema field names are the ONLY valid keys for meta-analytic records.

If the task requires JSON output:
- Output RAW JSON only - do NOT wrap in markdown code blocks (no ```json or ```).
- Do NOT include any explanatory text before or after the JSON.
- Output ONLY valid JSON that can be directly parsed.
- Ensure all record objects use exactly the schema-defined field names.
""",
            ),
            (
                "human",
                """**Task that was analyzed:** {task}

**Analyses from all participants:**

{all_results}

Produce the final, structured output. If a meta-analytic schema was provided, ensure all records strictly conform to its field names.
""",
            ),
        ]
    )
