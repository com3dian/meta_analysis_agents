"""
Execution topology definitions for the multi-agent system.

This module defines EXECUTION_TOPOLOGIES - how plans are executed, including:
- How many players work on each step in parallel
- How many debate rounds occur within each step
- Which player roles are available in the pool

Player configurations (PLAYER_CONFIGS) are defined in src/players/configs.py

Note: The orchestrator automatically adds 'relationship_analyst' to the player
pool when analyzing multi-table datasets. No need for separate multi-table topologies.
"""
from typing import Dict, Any


# ===================================================================
#  EXECUTION TOPOLOGY DEFINITIONS
# ===================================================================
# Define how plans should be executed, including:
# - players_per_step: How many players work on each step in parallel
# - debate_rounds: How many critique/revise cycles within each step
# - player_pool: Which player roles can be assigned to steps
#
# Note: For multi-table DataSources, 'relationship_analyst' is automatically
# added to the player pool by the orchestrator.

EXECUTION_TOPOLOGIES: Dict[str, Dict[str, Any]] = {
    "default": {
        "description": (
            "Standard execution with 3 parallel players per step, "
            "2 debate rounds, and comprehensive player pool."
        ),
        "players_per_step": 3,
        "debate_rounds": 2,
        "player_pool": ["data_analyst", "schema_expert", "metadata_specialist"],
    },
    "fast": {
        "description": (
            "Quick execution with 2 parallel players and minimal debate."
        ),
        "players_per_step": 2,
        "debate_rounds": 1,
        "player_pool": ["data_analyst", "schema_expert"],
    },
    "thorough": {
        "description": (
            "Thorough execution with more players and extended debate."
        ),
        "players_per_step": 4,
        "debate_rounds": 3,
        "player_pool": ["data_analyst", "schema_expert", "metadata_specialist", "critic"],
    },
    "single": {
        "description": (
            "Single player execution with no debate. Fastest but least robust."
        ),
        "players_per_step": 1,
        "debate_rounds": 0,
        "player_pool": ["data_analyst"],
    },
}
