"""
Defines the state schemas for the multi-agent system.

This module contains TypedDict definitions for:
1. StepExecutionState: State for executing a single plan step with parallel players
"""
from typing import TypedDict, List, Dict, Any, Optional, Type

from pydantic import BaseModel


class PlayerResult(TypedDict):
    """Result from a single player's execution."""
    player_name: str
    task: str
    tool_results: Dict[str, Any]
    analysis: str
    success: bool


class DebateEntry(TypedDict):
    """A single entry in the debate log."""
    round: int
    player_name: str
    entry_type: str  # 'initial_work', 'critique', 'revised_work'
    content: str


class StepExecutionState(TypedDict):
    """
    State for executing a single plan step with multiple parallel players.
    
    This state is used by the step-level debate graph where:
    1. Multiple players execute the same task in parallel
    2. Players debate (critique and revise) their results
    3. One of the players synthesizes the final result using their role expertise
    """
    # --- Step Configuration ---
    step_index: int                    # Index of this step in the plan
    task: str                          # The task description
    player_name: str                   # The player type for this step (from plan)
    rationale: str                     # Why this step is needed
    input_mappings: Dict[str, str]     # Maps param names to artifact names
    expected_outputs: List[str]        # Artifact names this step should produce
    target_resources: List[str]        # Which resources this step targets (empty = all)
    
    # --- Execution Context ---
    context_key: str                   # Key to registered ExecutionContext in tool registry
    context_info: Dict[str, Any]       # Serialized ExecutionContext info
    workspace: Dict[str, Any]          # Artifacts from previous steps
    
    # --- Player Configuration ---
    players: List[Any]                 # List of Player instances for this step
    synthesizer: Any                   # Player instance for synthesis (one of the players)
    
    # --- Debate Configuration ---
    max_debate_rounds: int             # Maximum debate rounds for this step
    current_debate_round: int          # Current debate round (starts at 1)
    
    # --- Dynamic State ---
    player_results: List[PlayerResult] # Results from parallel execution
    debate_log: List[DebateEntry]      # Log of debate entries
    
    # --- Structured Output ---
    output_schema: Optional[Type[BaseModel]]  # Pydantic schema for structured output
    
    # --- Output ---
    consolidated_result: Optional[Any] # Final synthesized result
    produced_artifacts: Dict[str, Any] # Artifacts produced by this step
    error: Optional[str]               # Error message if something went wrong
