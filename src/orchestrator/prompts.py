"""
This file stores all prompt templates for the multi-agent system.
"""
from langchain_core.prompts import ChatPromptTemplate


def get_planning_prompt() -> ChatPromptTemplate:
    """
    Returns the ChatPromptTemplate for the planning orchestrator.
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert planner agent.
Your goal is to generate a step-by-step plan to accomplish a given objective.

**Key Instructions:**
1.  **Think Step-by-Step**: Decompose the problem into a sequence of logical tasks.
2.  **Declare Data Dependencies**: Each step must declare its `inputs` and `outputs`.
    -   `inputs`: A dictionary mapping a task's required parameters to the names of artifacts created by previous steps. If a step needs no input from the workspace, this should be an empty dictionary.
    -   `outputs`: A list of new, unique artifact names that the step will create in the workspace.
3.  **Use Available Players**: You can only assign tasks to players from the provided list.
4.  **Provide Rationale**: Briefly explain the purpose of each step in the `rationale` field.

**Available Players:** 
{available_players}

**OUTPUT FORMAT (CRITICAL)**:
You MUST output **ONLY** a JSON object that conforms to the following schema:

{format_instructions}
""",
            ),
            (
                "human",
                "Generate a plan to achieve the following objective: '{objective}'.",
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
""",
            ),
            (
                "human",
                """**Task that was analyzed:** {task}

**Analyses from all participants:**

{all_results}

Produce the final, structured output.
""",
            ),
        ]
    )
