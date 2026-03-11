"""
Progress-aware orchestrator wrapper for Jupyter notebooks and terminals.

Wraps the core Orchestrator with tqdm progress bars that show:
- Plan generation phase
- Per-step progress (step index / total)
- Per-step sub-phase: executing → critiquing → revising → synthesizing

Uses tqdm.auto so the bars render as rich HTML widgets in Jupyter and
plain-text bars in a terminal.

Example usage in a notebook::

    from src.experimentutils import ProgressOrchestrator
    from src.standards import METADATA_STANDARDS

    orchestrator = ProgressOrchestrator(topology_name="default")
    result = orchestrator.run(source=context, objective=objective)
"""
from typing import Any, Dict, List, Optional, Type, Union

from tqdm.auto import tqdm

from src.orchestrator import Orchestrator
from src.context import ExecutionContext, create_context
from src.topology import EXECUTION_TOPOLOGIES


# Human-readable labels for each LangGraph node
_NODE_LABELS: Dict[str, str] = {
    "execute_parallel_node": "executing",
    "critique_node":         "critiquing",
    "revise_node":           "revising",
    "synthesize_node":       "synthesizing",
}


class ProgressOrchestrator:
    """
    Drop-in replacement for Orchestrator that displays tqdm progress bars.

    The outer bar tracks plan steps; the inner bar tracks the debate phase
    within each step (execute → critique → revise → synthesize).

    Args:
        topology_name: Execution topology (default, fast, thorough, single).
        **kwargs: Forwarded to Orchestrator (model_name, temperature, provider).
    """

    def __init__(self, topology_name: str = "default", **kwargs):
        self._orchestrator = Orchestrator(topology_name=topology_name, **kwargs)
        topology = EXECUTION_TOPOLOGIES[topology_name]
        self._debate_rounds = topology["debate_rounds"]
        self._players_per_step = topology["players_per_step"]

    @property
    def _phases_per_step(self) -> int:
        # execute(1) + [critique + revise] * rounds + synthesize(1)
        return 2 + self._debate_rounds * 2

    # ------------------------------------------------------------------
    # Public API — mirrors Orchestrator.run()
    # ------------------------------------------------------------------

    def run(
        self,
        source: Union[str, List[str], Dict[str, str], ExecutionContext],
        objective: str,
        name: str = "context",
        output_schema: Optional[Type[Any]] = None,
        **kwargs,
    ):
        """
        Generate a plan and execute it with live progress bars.

        Args:
            source: Paper path(s), directory, or existing ExecutionContext.
            objective: Extraction objective string (may embed schema).
            name: Context name (used when source is not an ExecutionContext).
            output_schema: Optional Pydantic model for structured final output.
            **kwargs: Forwarded to create_context when source is a path.

        Returns:
            ExecutionResult, or None if plan generation failed.
        """
        if isinstance(source, ExecutionContext):
            context = source
        else:
            context = create_context(source, name=name, **kwargs)

        # --- Phase 1: plan generation -----------------------------------
        with tqdm(total=1, desc="Generating plan", unit="plan", leave=True) as plan_bar:
            plan = self._orchestrator.generate_plan(
                context=context, objective=objective
            )
            plan_bar.update(1)

        if plan is None:
            print("Plan generation failed.")
            return None

        n_steps = len(plan.steps)

        # --- Phase 2: execution with nested progress bars ---------------
        step_bar = tqdm(
            total=n_steps,
            desc="Executing plan",
            unit="step",
            leave=True,
        )
        phase_bar = tqdm(
            total=self._phases_per_step,
            desc="  initializing",
            unit="phase",
            leave=False,
        )

        def on_step_start(step_idx: int, total: int, task: str, player: str):
            label = task if len(task) <= 55 else task[:52] + "..."
            step_bar.set_description(f"Step {step_idx + 1}/{total} [{player}]")
            step_bar.set_postfix_str(label)
            phase_bar.reset(total=self._phases_per_step)
            phase_bar.set_description("  → executing")

        def on_node_complete(step_idx: int, node_name: str):
            label = _NODE_LABELS.get(node_name, node_name)
            phase_bar.set_description(f"  → {label}")
            phase_bar.update(1)
            if node_name == "synthesize_node":
                step_bar.update(1)

        try:
            result = self._orchestrator.execute_plan(
                plan=plan,
                context=context,
                objective=objective,
                output_schema=output_schema,
                on_step_start=on_step_start,
                on_node_complete=on_node_complete,
            )
        finally:
            phase_bar.close()
            step_bar.close()

        return result


__all__ = ["ProgressOrchestrator"]
