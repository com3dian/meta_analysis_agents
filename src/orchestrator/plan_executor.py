"""
Plan Executor - Orchestrates the execution of a complete plan.

This module provides the PlanExecutor class that:
1. Takes a generated plan and execution topology
2. Iterates through each step sequentially
3. For each step, spawns parallel players and runs debates
4. Accumulates artifacts in a workspace
5. Produces the final result
"""
import logging
from typing import Dict, Any, List, Optional, Type

from pydantic import BaseModel

from src.core.schemas import Plan, ExecutionResult, StepResult
from src.context import ExecutionContext

from .step_executor import get_step_execution_graph, create_step_state
from .utils import filter_objective_by_sections
from ..topology import EXECUTION_TOPOLOGIES


class PlanExecutor:
    """
    Executes a complete plan using the specified topology.
    """

    def __init__(self, topology_name: str = "default"):
        """
        Initialize the PlanExecutor with a topology.

        Args:
            topology_name: Name of the execution topology to use
        """
        if topology_name not in EXECUTION_TOPOLOGIES:
            available = list(EXECUTION_TOPOLOGIES.keys())
            raise ValueError(
                f"Unknown topology '{topology_name}'. Available: {available}"
            )

        self.topology_name = topology_name
        self.topology = EXECUTION_TOPOLOGIES[topology_name]
        self.step_graph = get_step_execution_graph()

        logging.info(f"PlanExecutor initialized with topology: {topology_name}")
        logging.info(f"  Players per step: {self.topology['players_per_step']}")
        logging.info(f"  Debate rounds: {self.topology['debate_rounds']}")
        logging.info(f"  Player pool: {self.topology['player_pool']}")

    def execute(
        self,
        plan: Plan,
        context: ExecutionContext,
        context_key: str,
        output_schema: Optional[Type[BaseModel]] = None,
        player_pool: List[str] = None,
        objective: str = None,
    ) -> ExecutionResult:
        """
        Execute the complete plan.
        """
        effective_player_pool = player_pool or self.topology["player_pool"]

        logging.info("=" * 60)
        logging.info("STARTING PLAN EXECUTION")
        logging.info(f"Context: {context.name}")
        logging.info(f"Steps: {len(plan.steps)}")
        logging.info("=" * 60)

        workspace: Dict[str, Any] = {
            "_context_key": context_key,
            "_context_info": context.to_dict(),
        }
        
        # Store the full objective as an artifact so it can be filtered per step,
        # and also expose it (and the derived schema) as normal artifacts so that
        # they appear in the final workspace and can be referenced explicitly.
        if objective:
            workspace["_full_objective"] = objective
            workspace["initial_objective"] = objective
            
            try:
                schema_text = filter_objective_by_sections(
                    objective, ["META-ANALYTIC SCHEMA"]
                )
                workspace["meta_analytic_schema"] = schema_text
            except Exception as e:
                logging.warning(
                    f"Failed to extract meta-analytic schema from objective: {e}"
                )
        
        # Snapshot the original document content from the context as an artifact so
        # that the initial paper content is a first-class artifact in the plan.
        try:
            original_docs: Dict[str, Any] = {}
            for resource in context.resources:
                try:
                    content = context.read_resource(resource)
                    if isinstance(content, list):
                        content = "\n\n".join(str(p) for p in content)
                    original_docs[resource] = content
                except Exception as e:
                    logging.warning(
                        f"Error reading resource '{resource}' "
                        f"for original_document_text: {e}"
                    )
                    original_docs[resource] = f"[Error reading resource: {str(e)}]"
            
            workspace["original_document_text"] = original_docs
        except Exception as e:
            logging.warning(f"Failed to snapshot original document text: {e}")
        
        step_results: List[StepResult] = []

        plan_steps = plan.to_dict_list()

        for step_index, step_dict in enumerate(plan_steps):
            target_resources = step_dict.get("target_resources", [])

            logging.info("")
            logging.info(f"{'='*20} STEP {step_index + 1}/{len(plan_steps)} {'='*20}")
            logging.info(f"Task: {step_dict.get('task', 'Unknown')}")
            logging.info(f"Player: {step_dict.get('player', 'Unknown')}")
            logging.info(f"Rationale: {step_dict.get('rationale', 'None')}")
            if target_resources:
                logging.info(f"Target resources: {target_resources}")

            try:
                is_final_step = (step_index == len(plan_steps) - 1)
                step_output_schema = output_schema if is_final_step else None
                if step_output_schema:
                    logging.info(
                        f"  Final step will use structured output: {step_output_schema.__name__}"
                    )
                
                step_state = create_step_state(
                    step_index=step_index,
                    step_dict=step_dict,
                    context=context,
                    context_key=context_key,
                    workspace=workspace.copy(),
                    players_per_step=self.topology["players_per_step"],
                    debate_rounds=self.topology["debate_rounds"],
                    player_pool=effective_player_pool,
                    output_schema=step_output_schema,
                )

                final_step_state = self.step_graph.invoke(step_state)

                if final_step_state.get("error"):
                    error_msg = final_step_state["error"]
                    logging.error(f"Step {step_index + 1} failed: {error_msg}")
                    step_results.append(
                        StepResult(
                            step_index=step_index,
                            task=step_dict.get("task", ""),
                            player_role=step_dict.get("player", ""),
                            success=False,
                            error=error_msg,
                        )
                    )
                    continue

                produced_artifacts = final_step_state.get("produced_artifacts", {})
                workspace.update(produced_artifacts)

                step_results.append(
                    StepResult(
                        step_index=step_index,
                        task=step_dict.get("task", ""),
                        player_role=step_dict.get("player", ""),
                        individual_results=final_step_state.get("player_results", []),
                        debate_rounds_completed=final_step_state.get("current_debate_round", 0),
                        consolidated_result=final_step_state.get("consolidated_result", ""),
                        artifacts=produced_artifacts,
                        success=True,
                    )
                )

                logging.info(f"Step {step_index + 1} completed successfully")
                logging.info(f"  Artifacts produced: {list(produced_artifacts.keys())}")

            except Exception as e:
                error_msg = f"Unexpected error in step {step_index + 1}: {str(e)}"
                logging.error(error_msg)
                import traceback
                logging.error(traceback.format_exc())
                step_results.append(
                    StepResult(
                        step_index=step_index,
                        task=step_dict.get("task", ""),
                        player_role=step_dict.get("player", ""),
                        success=False,
                        error=error_msg,
                    )
                )

        successful_steps = sum(1 for r in step_results if r.success)
        overall_success = successful_steps == len(plan_steps)

        logging.info("")
        logging.info("=" * 60)
        logging.info("PLAN EXECUTION COMPLETE")
        logging.info(f"Steps completed: {successful_steps}/{len(plan_steps)}")
        logging.info(f"Overall success: {overall_success}")
        logging.info("=" * 60)

        return ExecutionResult(
            plan_steps_count=len(plan_steps),
            steps_completed=successful_steps,
            step_results=step_results,
            final_workspace=self._filter_workspace(workspace),
            final_output=self._extract_final_output(workspace),
            context_info=context.to_dict(),
            success=overall_success,
            error=None if overall_success else "Some steps failed",
        )

    def _filter_workspace(self, workspace: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out internal workspace keys."""
        return {k: v for k, v in workspace.items() if not k.startswith("_")}

    def _extract_final_output(self, workspace: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract the final output from the workspace.
        This is a placeholder and should be implemented based on the specific needs of the agent.
        """
        return self._filter_workspace(workspace)


def execute_plan(
    plan: Plan,
    context: ExecutionContext,
    context_key: str,
    meta_analytic_standard: str,
    topology_name: str = "default",
) -> ExecutionResult:
    """
    Convenience function to execute a plan.

    Args:
        plan: The Plan object to execute
        context: The ExecutionContext to analyze
        context_key: Key for the ExecutionContext in the tool registry
        meta_analytic_standard: The meta-analytic standard to follow
        topology_name: Name of the execution topology

    Returns:
        ExecutionResult with all results
    """
    executor = PlanExecutor(topology_name=topology_name)
    return executor.execute(
        plan=plan,
        context=context,
        context_key=context_key,
        meta_analytic_standard=meta_analytic_standard,
    )
