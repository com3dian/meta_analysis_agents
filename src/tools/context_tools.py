"""
Unified ExecutionContext Tools for the Multi-Agent System.
"""

from typing import Any, Dict

_context_registry: Dict[str, Any] = {}


def register_context(key: str, context: Any) -> str:
    """
    Register an ExecutionContext in the global registry.
    """
    _context_registry[key] = context
    return key


def get_context(key: str) -> Any:
    """Get an ExecutionContext from the registry."""
    if key not in _context_registry:
        raise KeyError(f"ExecutionContext '{key}' not found in registry")
    return _context_registry[key]


def clear_registry() -> None:
    """Clear all registered ExecutionContexts."""
    _context_registry.clear()

