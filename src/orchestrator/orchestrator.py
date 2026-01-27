"""
Main Orchestrator - Coordinates planning and execution.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Union

from langchain_core.output_parsers import PydanticOutputParser

from src.core.schemas import Plan, ExecutionResult

from ..config import DEFAULT_TOPOLOGY, LLM_PROVIDER, PLANNING_TEMPERATURE, create_llm
from ..context import ExecutionContext, create_context
from ..players import PLAYER_CONFIGS, Player, create_player_from_config
from ..tools.context_tools import clear_registry, register_context
from ..topology import EXECUTION_TOPOLOGIES
from .plan_executor import PlanExecutor
from .prompts import get_planning_prompt


class Orchestrator:
    """
    The main orchestrator that coordinates plan generation and execution.
    """

    def __init__(
        self,
        topology_name: str = None,
        model_name: str = None,
        temperature: float = None,
        provider: str = None,
    ):
        topology_name = topology_name or DEFAULT_TOPOLOGY
        temperature = temperature if temperature is not None else PLANNING_TEMPERATURE
        provider = provider or LLM_PROVIDER

        if topology_name not in EXECUTION_TOPOLOGIES:
            available = list(EXECUTION_TOPOLOGIES.keys())
            raise ValueError(
                f"Unknown topology '{topology_name}'. Available: {available}"
            )

        self.topology_name = topology_name
        self.topology = EXECUTION_TOPOLOGIES[topology_name]
        self.provider = provider

        self.llm = create_llm(
            model_name=model_name, temperature=temperature, provider=provider
        )
        self.parser = PydanticOutputParser(pydantic_object=Plan)
        self.prompt_template = get_planning_prompt()
        self.planning_chain = self.prompt_template | self.llm | self.parser

        self.executor = PlanExecutor(topology_name=topology_name)

        logging.info(f"Orchestrator initialized with topology: {topology_name}")

    def _get_effective_player_pool(self) -> list:
        return list(self.topology.get("player_pool", []))

    def _generate_player_manifest(self) -> str:
        player_pool = self._get_effective_player_pool()

        manifest_parts = []
        for role_name in player_pool:
            if role_name in PLAYER_CONFIGS:
                config = PLAYER_CONFIGS[role_name]
                player = create_player_from_config(config, name=role_name)
                manifest_parts.append(player.get_tool_manifest())

        return "\n\n".join(manifest_parts)

    def _generate_context_info(self, context: ExecutionContext) -> str:
        # This is a placeholder. In a real scenario, this would provide
        # rich information about the context for the planner.
        info_parts = [
            f"Context Name: {context.name}",
            f"Resources: {', '.join(context.resources)}",
        ]
        return "\n".join(info_parts)

    def generate_plan(
        self,
        context: ExecutionContext,
        objective: str,
    ) -> Optional[Plan]:
        logging.info("=" * 60)
        logging.info("GENERATING PLAN")
        logging.info(f"Context: {context.name}")
        logging.info(f"Objective: {objective}")
        logging.info("=" * 60)

        manifest = self._generate_player_manifest()
        context_info = self._generate_context_info(context)

        logging.info("Context info:")
        logging.info(context_info)
        logging.info("-" * 40)
        logging.info("Available players manifest")
        logging.info(manifest)
        logging.info("-" * 40)

        try:
            format_instructions = self.parser.get_format_instructions()

            prompt_inputs = {
                "objective": objective,
                "context_info": context_info,
                "available_players": manifest,
                "format_instructions": format_instructions,
            }

            generated_plan = self.planning_chain.invoke(prompt_inputs)

            logging.info("Plan generated successfully!")
            logging.info(f"Number of steps: {len(generated_plan.steps)}")
            for i, step in enumerate(generated_plan.steps):
                target_info = (
                    f" (resources: {step.target_resources})" if step.target_resources else ""
                )
                logging.info(
                    f"  Step {i + 1}: {step.task} (player: {step.player}){target_info}"
                )

            return generated_plan

        except Exception as e:
            logging.error(f"Plan generation failed: {e}")
            return None

    def execute_plan(
        self,
        plan: Plan,
        context: ExecutionContext,
    ) -> ExecutionResult:
        context_key = f"ctx_{uuid.uuid4().hex[:8]}"
        register_context(context_key, context)

        effective_player_pool = self._get_effective_player_pool()

        try:
            return self.executor.execute(
                plan=plan,
                context=context,
                context_key=context_key,
                player_pool=effective_player_pool,
            )
        finally:
            clear_registry()

    def run(
        self,
        source: Union[str, List[str], Dict[str, str], ExecutionContext],
        objective: str,
        name: str = "context",
        **kwargs,
    ) -> Optional[ExecutionResult]:
        if isinstance(source, ExecutionContext):
            context = source
        else:
            context = create_context(source, name=name, **kwargs)

        logging.info("=" * 60)
        logging.info("STARTING ORCHESTRATION")
        logging.info(f"Context: {context.name}")
        logging.info(f"Objective: {objective}")
        logging.info("=" * 60)

        plan = self.generate_plan(context=context, objective=objective)

        if plan is None:
            logging.error("Failed to generate plan. Aborting execution.")
            return None

        result = self.execute_plan(
            plan=plan, context=context
        )

        return result
