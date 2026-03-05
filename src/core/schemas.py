"""
Core Pydantic schemas for the multi-agent system.

This module defines the fundamental data structures:
1. Task: A single, executable unit of work.
2. Plan: A sequence of tasks with optional validation for data flow.
3. StepResult / ExecutionResult: Standard execution outputs for orchestrations.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, validator


class Task(BaseModel):
    """
    A single, executable step in the agent's plan.
    """

    task: str = Field(
        description="A specific and single task to be performed, e.g., 'get_row_count'."
    )
    player: str = Field(
        description="The name of the player role responsible for executing this task, e.g., 'data_analyst'."
    )
    rationale: str = Field(description="The reasoning for why this step is necessary.")
    target_resources: List[str] = Field(
        default_factory=list,
        description="List of resource names this step should operate on. Empty means all resources or a context-level operation.",
    )
    inputs: Dict[str, str] = Field(
        default_factory=dict,
        description="Maps a task's parameter names to the names of artifacts in the workspace that should be used as input.",
    )
    outputs: List[str] = Field(
        default_factory=list,
        description="A list of new artifact names that this step will produce and save to the workspace.",
    )
    objective_sections: List[str] = Field(
        default_factory=list,
        description="Optional: List of objective section names this step should see. If empty, the step sees the full objective. Sections like 'labeling', 'extraction', 'validation' can be used to filter the objective.",
    )


class Plan(BaseModel):
    """
    The complete, multi-step plan for the agent to execute.
    Includes validation to ensure logical consistency of task dependencies.
    """

    steps: List[Task] = Field(
        description="The list of sequential tasks the agent should follow."
    )

    @validator("steps")
    def validate_task_dependencies(cls, steps: List[Task]) -> List[Task]:
        """
        Best-effort validation that the inputs for each task are produced by a previous task.

        IMPORTANT:
        - This validator is **non-fatal**: it logs a warning on unmet dependencies
          but does NOT raise, to avoid hard failures on imperfect LLM plans.
        """
        produced_artifacts: Set[str] = set()

        for i, step in enumerate(steps):
            # Get the names of the artifacts required by the current step
            required_artifacts = set(step.inputs.values())

            # Check if all required artifacts have been produced by previous steps
            if not required_artifacts.issubset(produced_artifacts):
                missing = required_artifacts - produced_artifacts
                logging.warning(
                    "Plan validation warning: Step %d ('%s') has unmet dependencies. "
                    "Missing artifacts: %s",
                    i,
                    step.task,
                    missing,
                )

            # Add the artifacts produced by the current step to the set of available artifacts
            for output_artifact in step.outputs:
                produced_artifacts.add(output_artifact)

        return steps

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert plan steps to a list of dictionaries for state management."""
        return [step.model_dump() for step in self.steps]

    def pretty_print(self):
        print("Plan Steps:")
        for i, step in enumerate(self.steps):
            print(f"Step {i}: {step.task}")
            print(f"  Rationale: {step.rationale}")
            print(f"  Required Artifacts: {step.inputs}")
            print(f"  Produced Artifacts: {step.outputs}")


class StepResult(BaseModel):
    """
    The result of executing a single plan step.
    """

    step_index: int = Field(description="The index of the step in the plan.")
    task: str = Field(description="The task that was executed.")
    player_role: str = Field(description="The player role that executed this step.")

    individual_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Results from each player that worked on this step.",
    )
    debate_rounds_completed: int = Field(
        default=0, description="Number of debate rounds that were completed."
    )
    consolidated_result: Any = Field(
        default="", description="The synthesized result after debate."
    )
    artifacts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Artifacts produced by this step to add to workspace.",
    )
    success: bool = Field(
        default=True, description="Whether the step completed successfully."
    )
    error: Optional[str] = Field(
        default=None, description="Error message if the step failed."
    )


class ExecutionResult(BaseModel):
    """
    The complete result of executing a plan.
    """

    plan_steps_count: int = Field(description="Total number of steps in the plan.")
    steps_completed: int = Field(description="Number of steps successfully completed.")

    step_results: List[StepResult] = Field(
        default_factory=list, description="Results from each step."
    )
    final_workspace: Dict[str, Any] = Field(
        default_factory=dict,
        description="Final state of the workspace with all artifacts.",
    )
    final_output: Optional[Dict[str, Any]] = Field(
        default=None, description="The final result of the execution."
    )
    context_info: Dict[str, Any] = Field(
        default_factory=dict, description="Information about the ExecutionContext used."
    )
    success: bool = Field(
        default=True, description="Whether the entire plan executed successfully."
    )
    error: Optional[str] = Field(
        default=None, description="Error message if execution failed."
    )
