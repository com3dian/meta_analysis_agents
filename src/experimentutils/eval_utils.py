"""
Evaluation utilities for mapping papers to ground-truth annotations
and pre-processing paper text for better LLM extraction.
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .file_utils import _find_project_root, get_all_paper_paths

DEFAULT_ANNOTATION = "data/wopke_100/annotation/wopke100.xlsx"


def load_ground_truth(
    annotation_path: Optional[str] = None,
    sheet: str = "labels",
) -> pd.DataFrame:
    """
    Load the ground-truth annotation spreadsheet, deduplicating column names.

    Returns:
        DataFrame with unique column names (duplicate columns get a ``.N`` suffix).
    """
    import openpyxl

    if annotation_path is None:
        annotation_path = os.path.join(_find_project_root(), DEFAULT_ANNOTATION)

    wb = openpyxl.load_workbook(annotation_path, read_only=True, data_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    header = list(rows[0])

    seen: Dict[str, int] = {}
    deduped: List[str] = []
    for col in header:
        if col in seen:
            seen[col] += 1
            deduped.append(f"{col}.{seen[col]}")
        else:
            seen[col] = 0
            deduped.append(col)

    return pd.DataFrame(rows[1:], columns=deduped)


def build_study_paper_mapping(
    gt_df: Optional[pd.DataFrame] = None,
    paper_paths: Optional[List[str]] = None,
) -> Dict[int, str]:
    """
    Build a mapping from ``Study#`` in the annotation to the closest paper
    file path, using author last-name + publication year fuzzy matching.

    Returns:
        ``{study_id: paper_path}`` for every study that could be matched.
    """
    if gt_df is None:
        gt_df = load_ground_truth()
    if paper_paths is None:
        paper_paths = get_all_paper_paths()

    paper_stems = {Path(p).stem.lower(): p for p in paper_paths}

    studies: Dict[int, Tuple[str, str]] = {}
    for _, row in gt_df.iterrows():
        sid = row["Study#"]
        if sid in studies:
            continue
        author = str(row.get("Author", ""))
        year = str(row.get("Year of publication", ""))[:4]
        studies[int(sid)] = (author, year)

    mapping: Dict[int, str] = {}
    for sid, (author, year) in studies.items():
        last_name = author.split(",")[0].split(";")[0].strip().lower()
        if not last_name:
            continue
        exact, partial = [], []
        for stem, path in paper_stems.items():
            if last_name in stem:
                if year and year in stem:
                    exact.append(path)
                else:
                    partial.append(path)
        if exact:
            mapping[sid] = exact[0]
        elif partial:
            mapping[sid] = partial[0]

    return mapping


def get_paper_path_for_study(
    study_id: int,
    mapping: Optional[Dict[int, str]] = None,
) -> Optional[str]:
    """Return the paper path for a given ``Study#``, or *None* if unmatched."""
    if mapping is None:
        mapping = build_study_paper_mapping()
    return mapping.get(study_id)


# ---------------------------------------------------------------------------
# Text pre-processing: highlight tables & numbers
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(
    r"(?<!\w)"
    r"(\d[\d,]*\.?\d*)"           # integer or decimal, optionally with commas
    r"(\s*(?:"
    r"[%°]"                        # percent, degree
    r"|g\s*/\s*m2|g/m2"
    r"|kg\s*/?\s*ha|kg\s+ha"
    r"|t\s*/?\s*ha|t\s+ha"
    r"|Mg\s*/?\s*ha|Mg\s+ha"
    r"|bu\s*/?\s*acre"
    r"|plants?\s*/?\s*m2|plants\s+m2"
    r"|kg\s+N\s*/?\s*ha|kg\s+N\s+ha"
    r"|kg\s+P2O5\s*/?\s*ha"
    r"|kg\s+K2O\s*/?\s*ha"
    r"|days?"
    r"))?"
    r"(?!\w)",
    re.IGNORECASE,
)

_TABLE_HEADER_RE = re.compile(
    r"^(Table|TABLE|Fig(?:ure)?|FIGURE)\s+\d",
    re.MULTILINE,
)


def highlight_numbers_and_tables(text: str) -> str:
    """
    Annotate raw paper text with ``<<NUM: ... >>`` markers around numbers
    (with optional units) and ``<<TABLE_START>>`` / ``<<TABLE_END>>`` markers
    around table-like sections.

    This makes numerical data and tabular regions much more visible to LLMs
    that tend to overlook numbers embedded in dense text.
    """
    # Mark numbers
    def _mark_number(m: re.Match) -> str:
        full = m.group(0).strip()
        return f"<<NUM: {full} >>"

    text = _NUMBER_RE.sub(_mark_number, text)

    # Mark table-like blocks: consecutive lines with >= 3 numbers
    lines = text.split("\n")
    in_table = False
    out_lines: List[str] = []
    consecutive_numeric = 0

    for line in lines:
        num_count = len(re.findall(r"<<NUM:", line))
        is_table_header = bool(_TABLE_HEADER_RE.match(line.strip()))

        if is_table_header or num_count >= 3:
            if not in_table:
                out_lines.append("<<TABLE_START>>")
                in_table = True
            consecutive_numeric = 0
        elif in_table:
            consecutive_numeric += 1
            if consecutive_numeric > 2:
                out_lines.append("<<TABLE_END>>")
                in_table = False
                consecutive_numeric = 0

        out_lines.append(line)

    if in_table:
        out_lines.append("<<TABLE_END>>")

    return "\n".join(out_lines)


def build_extraction_context(
    paper_text: str,
    gt_df: Optional[pd.DataFrame] = None,
    study_id: Optional[int] = None,
) -> str:
    """
    Build an enriched extraction context by:
    1. Highlighting numbers and table regions.
    2. Appending a GT record-count hint (if available) so the LLM knows
       roughly how many records to expect.

    Args:
        paper_text: Raw text of the paper.
        gt_df: Ground-truth DataFrame (optional).
        study_id: Study# for this paper (optional).

    Returns:
        Annotated text ready to feed into extraction prompts.
    """
    enriched = highlight_numbers_and_tables(paper_text)

    if gt_df is not None and study_id is not None:
        gt_rows = gt_df[gt_df["Study#"] == study_id]
        if len(gt_rows) > 0:
            enriched += (
                f"\n\n<!-- HINT: the human annotation for this paper contains "
                f"{len(gt_rows)} experiment records. -->\n"
            )

    return enriched


__all__ = [
    "load_ground_truth",
    "build_study_paper_mapping",
    "get_paper_path_for_study",
    "highlight_numbers_and_tables",
    "build_extraction_context",
]
