"""
ExecutionContext Module - Unified context layer.

This module provides a unified interface for defining the "world" in which
the multi-agent system operates. This could be a traditional data source,
a file system, an API, etc.

The ExecutionContext abstraction allows the rest of the system to work with
the environment in a consistent way.

Quick Start:
    from src.context import create_context
    
    # Single paper
    ctx = create_context("./data/file_content.md")
    
    # Multiple papers
    ctx = create_context([
            "./data/file_1_content.md",
            "./data/file_2_content.md"
    ])
    
    # Use the ExecutionContext
    print(ctx.resources)
    df = ctx.read_resource("users")
    info = ctx.get_resource_info("users")
    schema = ctx.get_schema()
"""

from .base_context import (
    ExecutionContext,
    ContextType,
    ResourceInfo,
    FieldInfo,
    RelationshipInfo
)

from .context_factory import ContextFactory, create_context

__all__ = [
    # Base classes and models
    "ExecutionContext",
    "ContextType",
    "ResourceInfo",
    "FieldInfo",
    "RelationshipInfo",
    # Factory
    "ContextFactory",
    "create_context",
]