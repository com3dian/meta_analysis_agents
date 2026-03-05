"""
Deterministic XML tagging utilities and LangChain tool for the labeller.

This module provides a tool that:
- Reads raw document text from an ExecutionContext via context_key + resource
- Takes a list of (field, value) tuples
- Wraps each matching value in XML tags: <field>value</field>
"""

from typing import Any, List, Tuple, Type

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from .context_tools import get_context


def apply_xml_tags_to_content(
    content: str,
    field_value_pairs: List[Tuple[str, str]],
) -> str:
    """
    Deterministically apply XML tags to the content for each (field, value) pair.

    Very simple heuristic:
    - For each (field, value), replace all literal occurrences of `value`
      in the text with `<field>value</field>`.
    - Longer values are processed first to reduce nested/partial overlaps.
    """
    if not field_value_pairs:
        return content

    # Sort by value length descending to avoid shorter values tagging inside longer ones
    sorted_pairs = sorted(field_value_pairs, key=lambda fv: len(fv[1] or ""), reverse=True)

    tagged = content
    for field_name, value in sorted_pairs:
        if not field_name or not value:
            continue

        try:
            open_tag = f"<{field_name}>"
            close_tag = f"</{field_name}>"
            replacement = f"{open_tag}{value}{close_tag}"
            tagged = tagged.replace(value, replacement)
        except Exception:
            # Best-effort: on any unexpected error, skip this pair
            continue

    return tagged


class XMLTagFromFieldValuesInput(BaseModel):
    """Input schema for XMLTagFromFieldValuesTool."""

    context_key: str = Field(
        ...,
        description="Key for the ExecutionContext in the global context registry.",
    )
    resource: str = Field(
        ...,
        description="Logical resource name within the ExecutionContext to read.",
    )
    field_value_pairs: List[Tuple[str, str]] = Field(
        ...,
        description="List of (field, value) tuples to tag in the document text.",
    )


class XMLTagFromFieldValuesTool(BaseTool):
    """
    Tool that applies XML tags to a document based on a list of (field, value) tuples.

    - Reads the raw document content from the ExecutionContext
    - Wraps each occurrence of `value` in `<field>value</field>`
    - Returns the fully tagged document as a string
    """

    # Pydantic/LC v2: override BaseTool fields with type annotations
    name: str = "xml_tag_from_field_values"
    description: str = (
        "Apply XML tags to a document based on provided (field, value) tuples. "
        "Each occurrence of `value` in the text is wrapped as `<field>value</field>`."
    )
    args_schema: Type[BaseModel] = XMLTagFromFieldValuesInput

    def _run(
        self,
        context_key: str,
        resource: str,
        field_value_pairs: List[Tuple[str, str]],
        **_: Any,
    ) -> str:
        ctx = get_context(context_key)
        content = ctx.read_resource(resource)

        if isinstance(content, list):
            content = "\n\n".join(str(p) for p in content)
        else:
            content = str(content)

        return apply_xml_tags_to_content(content, field_value_pairs)

    async def _arun(
        self,
        context_key: str,
        resource: str,
        field_value_pairs: List[Tuple[str, str]],
        **_: Any,
    ) -> str:
        # Async not currently used in this project
        return self._run(context_key, resource, field_value_pairs)


def create_xml_tagging_tool() -> XMLTagFromFieldValuesTool:
    """Factory for the XML tagging tool."""
    return XMLTagFromFieldValuesTool()

