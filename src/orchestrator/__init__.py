"""
Orchestrator module for the multi-agent metadata extraction system.

This module provides:
- Orchestrator: Main class for planning and executing metadata extraction
- PlanExecutor: Executes a complete plan with parallel players
- Step execution via LangGraph

Uses the unified ExecutionContext abstraction for all data access.
See `src.context` for context creation and configuration.

Typical usage:
    from src.orchestrator import Orchestrator
    from src.standards import METANALYTIC_STANDARDS
    
    # Create orchestrator
    orchestrator = Orchestrator(topology_name="default")
    
    # Run on any data source (auto-detected)
    result = orchestrator.run(
        source="./data/file_content.md",  # Single file
        meta_analytic_standard=METANALYTIC_STANDARDS["climate_vs_cropyield"]
    )
    
    # Or multiple files
    result = orchestrator.run(
        source=[
            "./data/file_1_content.md",
            "./data/file_2_content.md"
        ],
        meta_analytic_standard=METANALYTIC_STANDARDS["climate_vs_cropyield"]
    )
    
    # Or directory of files
    result = orchestrator.run(
        source="./data/my_dataset/",
        meta_analytic_standard=METANALYTIC_STANDARDS["relational"]
    )
"""

from .orchestrator import Orchestrator
from .plan_executor import PlanExecutor, execute_plan
from src.core.schemas import Plan, Task, StepResult, ExecutionResult
from src.core.state import StepExecutionState

__all__ = [
    # Main classes
    "Orchestrator",
    "PlanExecutor",
    # Convenience functions
    "run_meta_analysis",
    "execute_plan",
    # Schema classes
    "Plan",
    "Task",
    "StepResult",
    "ExecutionResult",
    # State classes
    "StepExecutionState",
]
