"""
Step-level execution with parallel players and debate.

This module implements the core logic for executing a single plan step:
1. Spawn multiple players (based on topology)
2. Each player executes the task in parallel
3. Players debate (critique and revise) their results
4. A synthesizer consolidates the final result

The execution flow is:
    execute_parallel → critique → revise → [loop or synthesize]
"""
import json
import logging
import re
from typing import Dict, Any, List, Optional, Type

from pydantic import BaseModel

from langgraph.graph import StateGraph, END

from src.core.state import StepExecutionState, PlayerResult, DebateEntry
from ..players import Player, create_player_from_config, PLAYER_CONFIGS
from .utils import filter_objective_by_sections, get_labeling_objective


def _baseline_yield_record_count(workspace: Dict[str, Any]) -> Optional[int]:
    """
    Count rows in workspace['direct_records'] (yield_records / records) so record_extractor
    synthesis can enforce list cardinality. Returns None if unknown (e.g. markdown table).
    """
    dr = workspace.get("direct_records")
    if dr is None:
        return None

    def _len_from_dict(obj: Dict[str, Any]) -> Optional[int]:
        yr = obj.get("yield_records")
        if yr is None:
            yr = obj.get("records")
        if isinstance(yr, list):
            return len(yr)
        return None

    if isinstance(dr, dict):
        n = _len_from_dict(dr)
        if n is not None:
            return n

    if hasattr(dr, "model_dump"):
        try:
            dumped = dr.model_dump(by_alias=True)
            if isinstance(dumped, dict):
                n = _len_from_dict(dumped)
                if n is not None:
                    return n
        except Exception:
            pass

    if not isinstance(dr, str):
        return None

    s = dr.strip()
    if s.startswith("|") and "\n|" in s[:500]:
        return None

    def _try_parse(blob: str) -> Optional[int]:
        blob = blob.strip()
        try:
            parsed = json.loads(blob)
        except Exception:
            return None
        if isinstance(parsed, dict):
            return _len_from_dict(parsed)
        if isinstance(parsed, list) and parsed and all(isinstance(x, dict) for x in parsed):
            return len(parsed)
        return None

    n = _try_parse(s)
    if n is not None:
        return n

    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        n = _try_parse(m.group(0))
        if n is not None:
            return n

    m = re.search(r"\[[\s\S]*\]", s)
    if m:
        n = _try_parse(m.group(0))
        if n is not None:
            return n

    return None


# ===================================================================
#  NODE FUNCTIONS
# ===================================================================

def execute_parallel_node(state: StepExecutionState) -> Dict[str, Any]:
    """
    Execute the task with all players in parallel.
    """
    logging.info(f"--- STEP {state['step_index']}: PARALLEL EXECUTION ---")
    logging.info(f"Task: {state['task']}")
    logging.info(f"Players: {len(state['players'])}")
    
    players: List[Player] = state["players"]
    task = state["task"]
    context_key = state["context_key"]
    context_info = state["context_info"]
    target_resources = state.get("target_resources", [])
    workspace = state["workspace"]
    input_mappings = state["input_mappings"]
    
    player_results: List[PlayerResult] = []
    initial_debate_entries: List[DebateEntry] = []
    
    for player in players:
        try:
            result = player.execute_task(
                task=task,
                context_key=context_key,
                context_info=context_info,
                workspace=workspace,
                inputs=input_mappings,
                target_resources=target_resources
            )
            
            player_results.append({
                "player_name": player.name,
                "task": task,
                "tool_results": result.get("tool_results", {}),
                "analysis": result.get("analysis", ""),
                "success": result.get("success", True)
            })
            
            initial_debate_entries.append({
                "round": 1,
                "player_name": player.name,
                "entry_type": "initial_work",
                "content": result.get("analysis", "")
            })
            
            logging.info(f"  Player '{player.name}' completed execution")
            
        except Exception as e:
            logging.error(f"  Player '{player.name}' failed: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            player_results.append({
                "player_name": player.name,
                "task": task,
                "tool_results": {},
                "analysis": f"Error: {str(e)}",
                "success": False
            })
    
    return {
        "player_results": player_results,
        "debate_log": state.get("debate_log", []) + initial_debate_entries,
        "current_debate_round": 1
    }


def critique_node(state: StepExecutionState) -> Dict[str, Any]:
    """
    Each player critiques the work of other players.
    """
    current_round = state["current_debate_round"]
    logging.info(f"--- STEP {state['step_index']}: CRITIQUE (Round {current_round}) ---")
    
    players: List[Player] = state["players"]
    task = state["task"]
    debate_log = state["debate_log"]
    
    current_round_work = {
        entry["player_name"]: entry["content"]
        for entry in debate_log
        if entry["round"] == current_round and "work" in entry["entry_type"]
    }
    
    new_entries: List[DebateEntry] = []
    
    for player in players:
        other_work = {k: v for k, v in current_round_work.items() if k != player.name}
        
        if not other_work:
            continue
            
        try:
            critique = player.critique_work(task=task, other_players_work=other_work)
            new_entries.append({
                "round": current_round,
                "player_name": player.name,
                "entry_type": "critique",
                "content": critique
            })
            logging.info(f"  Player '{player.name}' provided critique")
        except Exception as e:
            logging.error(f"  Player '{player.name}' critique failed: {str(e)}")
    
    return {"debate_log": debate_log + new_entries}


def revise_node(state: StepExecutionState) -> Dict[str, Any]:
    """
    Each player revises their work based on critiques.
    """
    current_round = state["current_debate_round"]
    next_round = current_round + 1
    logging.info(f"--- STEP {state['step_index']}: REVISION (Round {next_round}) ---")
    
    players: List[Player] = state["players"]
    task = state["task"]
    debate_log = state["debate_log"]
    
    critiques = [
        entry["content"]
        for entry in debate_log
        if entry["round"] == current_round and entry["entry_type"] == "critique"
    ]
    
    new_entries: List[DebateEntry] = []
    updated_results: List[PlayerResult] = []
    
    for player in players:
        original_work = next(
            (entry["content"] for entry in debate_log
             if entry["player_name"] == player.name 
             and "work" in entry["entry_type"]
             and entry["round"] == current_round),
            ""
        )
        
        try:
            revised = player.revise_work(
                task=task,
                my_original_work=original_work,
                critiques=critiques
            )
            new_entries.append({
                "round": next_round,
                "player_name": player.name,
                "entry_type": "revised_work",
                "content": revised
            })
            
            updated_results.append({
                "player_name": player.name,
                "task": task,
                "tool_results": {},
                "analysis": revised,
                "success": True
            })
            
            logging.info(f"  Player '{player.name}' revised their work")
        except Exception as e:
            logging.error(f"  Player '{player.name}' revision failed: {str(e)}")
    
    return {
        "debate_log": debate_log + new_entries,
        "current_debate_round": next_round,
        "player_results": updated_results if updated_results else state["player_results"]
    }


def synthesize_node(state: StepExecutionState) -> Dict[str, Any]:
    """
    Synthesize all player results into a consolidated output.
    """
    logging.info(f"--- STEP {state['step_index']}: SYNTHESIS ---")
    
    synthesizer: Player = state["synthesizer"]
    task = state["task"]
    player_results = state["player_results"]
    expected_outputs = state["expected_outputs"]
    output_schema: Optional[Type[BaseModel]] = state.get("output_schema")
    
    if output_schema:
        logging.info(f"  Using structured output with schema: {output_schema.__name__}")
    
    try:
        results_for_synthesis = [
            {
                "player": r["player_name"],
                "analysis": r["analysis"],
                "tool_results": r["tool_results"]
            }
            for r in player_results
        ]
        
        # Include schema from workspace if available for synthesis
        workspace = state.get("workspace", {})
        synth_insert_pos = 0
        if "meta_analytic_schema" in workspace:
            schema_content = workspace["meta_analytic_schema"]
            # Add schema as a special entry in results for synthesis
            results_for_synthesis.insert(0, {
                "player": "schema_reference",
                "analysis": f"**META-ANALYTIC SCHEMA (CRITICAL - USE THESE EXACT FIELD NAMES):**\n{schema_content}\n\n**IMPORTANT**: All meta-analytic records MUST use exactly these field names. Do NOT invent new field names.",
                "tool_results": {}
            })
            synth_insert_pos = 1

        baseline_record_count: Optional[int] = None
        if synthesizer is not None and synthesizer.name == "record_extractor":
            baseline_record_count = _baseline_yield_record_count(workspace)
            if baseline_record_count is not None and baseline_record_count > 0:
                results_for_synthesis.insert(synth_insert_pos, {
                    "player": "record_cardinality_constraint",
                    "analysis": (
                        f"**NON-NEGOTIABLE:** `direct_records` baseline has **{baseline_record_count}** "
                        "experiment rows. The final structured record list MUST have **at least** that "
                        "many elements, same baseline order for those positions (refine in place). "
                        "Do not merge into one summary record."
                    ),
                    "tool_results": {},
                })

        consolidated = synthesizer.synthesize_results(
            task=task,
            all_results=results_for_synthesis,
            output_schema=output_schema,
            baseline_record_count=baseline_record_count,
        )
        
        if output_schema and isinstance(consolidated, BaseModel):
            artifact_value = consolidated.model_dump(by_alias=True)
        else:
            artifact_value = consolidated
        
        produced_artifacts = {}
        for output_name in expected_outputs:
            produced_artifacts[output_name] = artifact_value
        
        logging.info(f"  Synthesis complete. Produced artifacts: {list(produced_artifacts.keys())}")
        if isinstance(artifact_value, dict):
            preview = str(artifact_value)[:200].replace('\n', ' ')
        else:
            preview = str(artifact_value)[:200].replace('\n', ' ')
        if len(str(artifact_value)) > 200:
            preview += "..."
        logging.info(f"    Synthesized output: {preview}")
        
        return {
            "consolidated_result": consolidated,
            "produced_artifacts": produced_artifacts
        }
        
    except Exception as e:
        error_msg = f"Synthesis failed: {str(e)}"
        logging.error(f"  {error_msg}")
        return {
            "error": error_msg,
            "consolidated_result": None,
            "produced_artifacts": {}
        }


def debate_router(state: StepExecutionState) -> str:
    """
    Decide whether to continue debate or synthesize.
    """
    if state.get("error"):
        logging.error(f"Error detected, ending step: {state['error']}")
        return "__end__"
    
    current_round = state["current_debate_round"]
    max_rounds = state["max_debate_rounds"]
    num_players = len(state["players"])
    
    if num_players <= 1:
        return "synthesize_node"
    
    if current_round < max_rounds:
        return "critique_node"
    else:
        return "synthesize_node"


# ===================================================================
#  GRAPH CONSTRUCTION
# ===================================================================

def get_step_execution_graph():
    """
    Constructs and compiles the StateGraph for step execution.
    """
    graph = StateGraph(StepExecutionState)
    
    graph.add_node("execute_parallel_node", execute_parallel_node)
    graph.add_node("critique_node", critique_node)
    graph.add_node("revise_node", revise_node)
    graph.add_node("synthesize_node", synthesize_node)
    
    graph.set_entry_point("execute_parallel_node")
    
    graph.add_conditional_edges(
        "execute_parallel_node",
        debate_router,
        {
            "critique_node": "critique_node",
            "synthesize_node": "synthesize_node",
            "__end__": END
        }
    )
    
    graph.add_edge("critique_node", "revise_node")
    
    graph.add_conditional_edges(
        "revise_node",
        debate_router,
        {
            "critique_node": "critique_node",
            "synthesize_node": "synthesize_node",
            "__end__": END
        }
    )
    
    graph.add_edge("synthesize_node", END)
    
    return graph.compile()


# ===================================================================
#  HELPER FUNCTIONS
# ===================================================================

def create_step_state(
    step_index: int,
    step_dict: Dict[str, Any],
    context: Any,
    context_key: str,
    workspace: Dict[str, Any],
    players_per_step: int,
    debate_rounds: int,
    player_pool: List[str],
    output_schema: Optional[Type[BaseModel]] = None
) -> StepExecutionState:
    """
    Create the initial state for executing a step.
    """
    # The plan assigns a specific player role to each step — use it.
    # Fall back to pool[0] only if the assigned role is missing from PLAYER_CONFIGS.
    player_name = step_dict.get("player", "")

    players = []
    if player_name and player_name in PLAYER_CONFIGS:
        # Primary player: the role the planner chose for this step.
        config = PLAYER_CONFIGS[player_name]
        players.append(create_player_from_config(config, name=player_name))
        # If more players are requested (debate topology), draw the rest from the pool.
        for i in range(1, players_per_step):
            role_name = player_pool[i % len(player_pool)]
            config = PLAYER_CONFIGS.get(role_name, {})
            players.append(create_player_from_config(config, name=f"{role_name}_{i+1}"))
    else:
        # Fallback: planner gave an unknown/missing role — use the pool as before.
        logging.warning(
            f"Step {step_dict.get('task','?')!r}: assigned player {player_name!r} "
            f"not found in PLAYER_CONFIGS; falling back to player pool."
        )
        for i in range(min(players_per_step, len(player_pool))):
            role_name = player_pool[i % len(player_pool)]
            config = PLAYER_CONFIGS.get(role_name, {})
            players.append(create_player_from_config(config, name=f"{role_name}_{i+1}"))

    # The synthesizer consolidates results; use the assigned player so the same
    # expertise is applied during synthesis as during execution.
    synthesizer = players[0] if players else None
    
    target_resources = step_dict.get("target_resources", [])
    
    return StepExecutionState(
        step_index=step_index,
        task=step_dict.get("task", ""),
        player_name=player_name,
        rationale=step_dict.get("rationale", ""),
        input_mappings=step_dict.get("inputs", {}),
        expected_outputs=step_dict.get("outputs", []),
        target_resources=target_resources,
        context_key=context_key,
        context_info=context.to_dict(),
        workspace=workspace,
        players=players,
        synthesizer=synthesizer,
        max_debate_rounds=debate_rounds,
        current_debate_round=0,
        player_results=[],
        debate_log=[],
        output_schema=output_schema,
        consolidated_result=None,
        produced_artifacts={},
        error=None,
    )
