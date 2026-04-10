"""
Deterministic XML tagging utilities and LangChain tools for the labeller stages.

Two tools are provided:
- xml_tag_from_field_values: wraps field values with <FieldName> tags (Step 2)
- xml_tag_records: wraps table rows with <Record_N> tags (Step 4)
"""

from typing import Any, Dict, List, Optional, Tuple, Type

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
    """Factory for the field-value XML tagging tool."""
    return XMLTagFromFieldValuesTool()


# ── Record-level XML tagging ──────────────────────────────────────────────────

_DEFAULT_DISCRIMINATING_FIELDS = [
    "unified yield ic 1",
    "unified yield ic 2",
    "unified yield sc 1",
    "unified yield sc 2",
    "Density ic 1",
    "Density ic 2",
]


def apply_record_tags_to_content(
    labeled_text: str,
    candidate_records: List[Dict[str, Any]],
    discriminating_fields: Optional[List[str]] = None,
) -> str:
    """
    Wrap lines of *labeled_text* with ``<Record_N>...</Record_N>`` tags based on
    the treatment-specific field values in *candidate_records*.

    Algorithm
    ---------
    For each candidate record the function collects its non-null values for every
    *discriminating_field*.  It then scans every line in the text and counts how
    many of those values appear in that line (matching the raw value **or** the
    XML-tagged form ``<field>value</field>``).  The line with the highest match
    count is selected and wrapped.  Ties are broken in favour of the first match.

    A line that already has the best match for record A is not reassigned to
    record B later — first-writer wins.

    Parameters
    ----------
    labeled_text:
        The field-tagged document text produced by the ``xml_tag_from_field_values``
        step.
    candidate_records:
        List of record dicts.  Each must contain at least ``record_index`` (int)
        and the relevant field values.
    discriminating_fields:
        Field names whose values differ between records and therefore uniquely
        identify a row.  Defaults to yield + density fields.

    Returns
    -------
    str
        Full document text with ``<Record_N>...</Record_N>`` wrappers added to
        the best-matching line for each record.  All existing XML tags are
        preserved unchanged.
    """
    if discriminating_fields is None:
        discriminating_fields = _DEFAULT_DISCRIMINATING_FIELDS

    lines = labeled_text.split("\n")
    # line_idx -> (record_index, match_count) — first-writer wins on ties
    line_best: Dict[int, Tuple[int, int]] = {}

    for rec_idx, record in enumerate(candidate_records, start=1):
        record_index = int(record.get("record_index") or rec_idx)

        # Collect non-empty discriminating values for this record
        search_values: List[str] = []
        for fname in discriminating_fields:
            val = record.get(fname)
            if val is not None and str(val).strip() not in ("", "None", "null"):
                search_values.append(str(val).strip())

        if not search_values:
            continue

        # Score each line by number of matching values
        best_line_idx: Optional[int] = None
        best_count = 0
        for line_idx, line in enumerate(lines):
            count = sum(
                1
                for v in search_values
                if v in line or f">{v}<" in line
            )
            if count > best_count:
                best_count = count
                best_line_idx = line_idx

        if best_line_idx is None or best_count == 0:
            continue

        # First-writer wins: only claim a line if not already taken with a
        # higher match count
        existing = line_best.get(best_line_idx)
        if existing is None or best_count > existing[1]:
            line_best[best_line_idx] = (record_index, best_count)

    # Apply wrappers
    tagged_lines = list(lines)
    for line_idx, (record_index, _) in line_best.items():
        tagged_lines[line_idx] = (
            f"<Record_{record_index}>"
            f"{tagged_lines[line_idx]}"
            f"</Record_{record_index}>"
        )

    return "\n".join(tagged_lines)


class XMLTagRecordsInput(BaseModel):
    """Input schema for XMLTagRecordsTool."""

    labeled_text: str = Field(
        ...,
        description="The field-XML-tagged document text produced by the field-value labeller.",
    )
    candidate_records: List[Dict[str, Any]] = Field(
        ...,
        description=(
            "List of candidate record dicts.  Each must have 'record_index' (int) "
            "and field-value entries whose keys match the discriminating_fields."
        ),
    )
    discriminating_fields: Optional[List[str]] = Field(
        default=None,
        description=(
            "Field names whose values differ between records and can uniquely identify "
            "a table row.  Defaults to yield and density fields."
        ),
    )


class XMLTagRecordsTool(BaseTool):
    """
    Tool that wraps table rows with ``<Record_N>`` XML tags.

    Takes the field-tagged document from the previous labeller step and a list
    of candidate records, then deterministically labels each row/line in the
    document that best matches each record's discriminating field values.
    """

    name: str = "xml_tag_records"
    description: str = (
        "Wrap rows in a field-tagged document with <Record_N>...</Record_N> tags. "
        "Each candidate record is matched to the document line that contains the "
        "most of its discriminating field values (yield/density)."
    )
    args_schema: Type[BaseModel] = XMLTagRecordsInput

    def _run(
        self,
        labeled_text: str,
        candidate_records: List[Dict[str, Any]],
        discriminating_fields: Optional[List[str]] = None,
        **_: Any,
    ) -> str:
        return apply_record_tags_to_content(
            labeled_text=labeled_text,
            candidate_records=candidate_records,
            discriminating_fields=discriminating_fields,
        )

    async def _arun(
        self,
        labeled_text: str,
        candidate_records: List[Dict[str, Any]],
        discriminating_fields: Optional[List[str]] = None,
        **_: Any,
    ) -> str:
        return self._run(labeled_text, candidate_records, discriminating_fields)


def create_xml_records_tagging_tool() -> XMLTagRecordsTool:
    """Factory for the record-level XML tagging tool."""
    return XMLTagRecordsTool()

