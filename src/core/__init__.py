"""
Core module for the multi-agent meta-analysis system.

Contains:
- schemas: Pydantic models for Plan, Task, StepResult, ExecutionResult
- state: TypedDict definitions for step execution state
- schema_factory: Dynamic Pydantic model generation from metadata standards
"""

from .schemas import Plan, Task, StepResult, ExecutionResult
from .state import StepExecutionState, PlayerResult, DebateEntry
from .schema_factory import SchemaFactory, create_output_schema

__all__ = [
    # Schemas
    "Plan",
    "Task",
    "StepResult",
    "ExecutionResult",
    # State
    "StepExecutionState",
    "PlayerResult",
    "DebateEntry",
    # Schema Factory
    "SchemaFactory",
    "create_output_schema",
]
