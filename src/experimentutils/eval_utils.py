"""
Evaluation utilities for mapping papers to ground-truth annotations
and pre-processing paper text for better LLM extraction.

Public API
----------
Field-level scoring
    field_similarity_score  – hybrid numeric / year / ROUGE-L scorer

Record matching & evaluation
    find_crop_swap_pairs    – detect all crop-1/crop-2 column pairs
    match_records_greedy    – one-to-one greedy matching with swap support
    evaluate_method_scores  – normalized score over all GT records
    print_matching_pairs    – pretty-print matched pairs for inspection
    print_field_value_pairs_vs_gt – Step-1 pairs vs GT (same column layout)
"""
import ast
import json
import os
import re
import decimal
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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


# ---------------------------------------------------------------------------
# Field-level similarity scoring
# ---------------------------------------------------------------------------

_MISSING_TOKENS = {"", "nan", "none", "null", "n/a", "na"}

# Matches a 4-digit calendar year (1000–2099)
_YEAR_RE = re.compile(r"\b(1\d{3}|20\d{2})\b")

# Matches a value that is *purely* numeric (optional leading sign, comma
# thousands separators, and a single decimal point)
_PURE_NUMBER_RE = re.compile(r"^[+-]?[\d,]+\.?\d*$")

# Matches a parenthetical scientific name, e.g. "(Vicia faba)" or "(Triticum aestivum L.)"
_SCIENTIFIC_NAME_RE = re.compile(r"\s*\([A-Z][a-z]+(?:\s+[a-z]+\.?)+\)")


def _is_missing(v: Any) -> bool:
    """Return True for NaN, None, or any blank/null-like string."""
    if v is None:
        return True
    return str(v).strip().lower() in _MISSING_TOKENS


def _try_parse_number(v: Any) -> Optional[float]:
    """Return float if *v* represents a pure number, else None."""
    s = str(v).strip().replace(",", "")
    if _PURE_NUMBER_RE.match(s):
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _try_parse_decimal(v: Any) -> Optional["decimal.Decimal"]:
    """
    Return a Decimal if *v* is a pure number, else None.

    Uses the string form (after removing comma separators) to avoid float
    rounding artifacts during exact-equality checks.
    """
    s = str(v).strip().replace(",", "")
    if not _PURE_NUMBER_RE.match(s):
        return None
    try:
        # Decimal accepts leading +/-, integer and decimal forms.
        return decimal.Decimal(s)
    except (decimal.InvalidOperation, ValueError):
        return None


def _remove_decimal_point_numeric_string(v: Any) -> Optional[str]:
    """
    If *v* is a pure number, return a canonical string with the decimal point removed.

    Example:
        "3.124" -> "3124"
        "3124"  -> "3124"
        "-3.10" -> "-310"
    """
    s = str(v).strip().replace(",", "")
    if not _PURE_NUMBER_RE.match(s):
        return None
    return s.replace(".", "")


def _excel_serial_to_iso(v: Any) -> Optional[str]:
    """
    Convert an Excel-style serial day (1 = 1900-01-01) into an ISO date string.

    Only treats values within a reasonable range as serial dates to avoid
    misinterpreting arbitrary integers.
    """
    s = str(v).strip().replace(",", "")
    # Must look like a pure number (integer or simple decimal)
    if not _PURE_NUMBER_RE.match(s):
        return None
    try:
        dec = decimal.Decimal(s)
    except (decimal.InvalidOperation, ValueError):
        return None

    # Only treat integer-valued decimals as serials (e.g. 35693 or 35693.0)
    try:
        n = int(dec)
    except (OverflowError, ValueError):
        return None

    # Rough guard rails: years roughly between 1950 and 2050
    # 1900-01-01 + 20000 days ≈ 1954, +60000 ≈ 2064
    if n < 20000 or n > 60000:
        return None
    base = date(1900, 1, 1)
    # Empirically, Excel-style serials in the Wopke sheets decode one day
    # ahead with the naive 1900-01-01 + (n-1) formula, so we subtract one
    # extra day to align with the GT dates.
    d = base + timedelta(days=n - 2)
    return d.isoformat()


def _normalize_text(v: Any) -> str:
    """
    Prepare a text value for soft token comparison:

    Lowercase and collapse whitespace while preserving all content, including
    parenthetical scientific names.
    """
    return " ".join(str(v).lower().split())


def _soft_tokens_match(t1: str, t2: str) -> bool:
    """
    Return ``True`` when two tokens are considered equivalent.

    Exact match OR one is a prefix of the other, which handles common
    plural/singular crop-name variants (``"beans"`` vs ``"bean"``,
    ``"wheats"`` vs ``"wheat"``, ``"maizes"`` vs ``"maize"``).
    """
    print(f"Soft tokens match: {t1} vs {t2}")
    return t1 == t2 or t1.startswith(t2) or t2.startswith(t1)


def _extract_year(v: Any) -> Optional[str]:
    """
    Return the first 4-digit year found in *v*, or None.

    Handles plain years ("1987"), slash ranges ("1987/88"),
    and hyphen ranges ("1987-88", "1987-1988").
    """
    m = _YEAR_RE.search(str(v))
    return m.group(1) if m else None


def rouge_l_soft_score(ref_val: Any, hyp_val: Any) -> float:
    """
    Compute ROUGE-L F1 on character sequences.

    This helper is exposed so ROUGE-L behavior can be tested directly without
    running the full field-type routing in ``field_similarity_score``.
    """
    ref_chars = [c for c in _normalize_text(ref_val) if not c.isspace()]
    hyp_chars = [c for c in _normalize_text(hyp_val) if not c.isspace()]
    if not ref_chars or not hyp_chars:
        return 0.0

    m, nl = len(ref_chars), len(hyp_chars)
    dp = [[0] * (nl + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, nl + 1):
            dp[i][j] = (
                dp[i - 1][j - 1] + 1
                if ref_chars[i - 1] == hyp_chars[j - 1]
                else max(dp[i - 1][j], dp[i][j - 1])
            )

    lcs = dp[m][nl]
    p, r = lcs / nl, lcs / m
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def field_similarity_score(
    ref_val: Any,
    hyp_val: Any,
    field_name: Optional[str] = None) -> float:
    """
    Hybrid similarity score for a single extracted field value compared to
    its ground-truth counterpart.

    The function selects a comparison strategy automatically based on the
    detected value type:

    1. **Missing** – either side is ``None``, ``NaN``, empty, or a null-like
       string (``"nan"``, ``"none"``, ``"null"``, ``"n/a"``).  Returns ``0.0``
       so jointly-missing fields do not receive a free perfect score.

    2. **Numeric** – both sides parse as pure numbers (integers or decimals,
       optionally with comma thousands-separators).

       - Exact numeric match → ``1.0``
       - Else, if removing the decimal point makes the strings equal
         (e.g. ``"3.124"`` ↔ ``"3124"``) → ``0.5``
       - Else → ``0.0``

       No approximate / tolerance / relative-error scoring is used.

    3. **Year** – both sides contain a 4-digit calendar year.  Normalises
       year-range strings (``"1987-88"``, ``"1987/88"``) to their first
       4-digit year before comparing.  Returns ``1.0`` for equal years,
       ``0.0`` otherwise.

    4. **Text / categorical** – ROUGE-L F1 with two pre-processing steps
       applied before tokenisation:

       * **Scientific name stripping** – removes parenthetical Latin binomials
         so ``"field beans (Vicia faba)"`` becomes ``"field beans"``.
       * **Prefix / stem token matching** – two tokens are treated as equal
         when one is a prefix of the other, handling common plural/singular
         variants (``"beans"`` ↔ ``"bean"``, ``"wheats"`` ↔ ``"wheat"``).

    Args:
        ref_val: Reference (ground-truth) value – any scalar type.
        hyp_val: Hypothesis (extracted) value – any scalar type.

    Returns:
        Similarity score in ``[0.0, 1.0]``.

    Examples:
        >>> field_similarity_score("3.67", "3.7")
        0.0
        >>> field_similarity_score("3.124", "3124")
        0.5
        >>> field_similarity_score("1987-88", "1987")
        1.0
        >>> field_similarity_score("Bean", "bean")
        1.0
        >>> field_similarity_score("field beans (Vicia faba)", "Bean")
        0.667...
        >>> field_similarity_score("wheat (Triticum aestivum)", "Wheat")
        1.0
        >>> field_similarity_score("Bean", "Wheat")
        0.0
        >>> field_similarity_score(None, "1987")
        0.0
        >>> field_similarity_score(float("nan"), float("nan"))
        0.0
    """
    if _is_missing(ref_val) or _is_missing(hyp_val):
        return 0.0

    # ── 1. Numeric comparison ────────────────────────────────────────────────
    ref_dec = _try_parse_decimal(ref_val)
    hyp_dec = _try_parse_decimal(hyp_val)
    if ref_dec is not None and hyp_dec is not None:
        print(f"Numeric comparison: {ref_dec} vs {hyp_dec}")
        # Exact numeric equality first (e.g. "3" == "3.0" == 3)
        if ref_dec == hyp_dec:
            return 1.0
        # Second-chance: match by removing the decimal point (e.g. 3.124 ~ 3124)
        ref_no_dot = _remove_decimal_point_numeric_string(ref_val)
        hyp_no_dot = _remove_decimal_point_numeric_string(hyp_val)
        if ref_no_dot is not None and hyp_no_dot is not None and ref_no_dot == hyp_no_dot:
            return 0.5
        return 0.0

    # ── 2. Excel-serial-style date comparison ───────────────────────────────
    # If one or both values look like Excel serial days and can be converted
    # to calendar dates, compare their ISO strings for exact match.
    ref_excel = _excel_serial_to_iso(ref_val)
    hyp_excel = _excel_serial_to_iso(hyp_val)
    if ref_excel is not None or hyp_excel is not None:
        print(f"Excel-serial-style date comparison: {ref_excel} vs {hyp_excel}")
        # Try to normalise the other side to ISO date as well:
        # - if it's already ISO-like (YYYY-MM-DD), accept as-is;
        # - otherwise, fall back to year-only comparison below.
        def _normalize_dateish(v: Any) -> Optional[str]:
            s = str(v).strip()
            # Simple ISO date check
            if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
                return s
            return _excel_serial_to_iso(s)

        ref_iso = ref_excel or _normalize_dateish(ref_val)
        hyp_iso = hyp_excel or _normalize_dateish(hyp_val)
        if ref_iso is not None and hyp_iso is not None:
            return 1.0 if ref_iso == hyp_iso else 0.0

    # ── 3. Year comparison ───────────────────────────────────────────────────
    ref_year = _extract_year(ref_val)
    hyp_year = _extract_year(hyp_val)
    if ref_year is not None and hyp_year is not None:
        print(f"Year comparison: {ref_year} vs {hyp_year}")
        return 1.0 if ref_year == hyp_year else 0.0

    # ── 4. Text / categorical comparison ────────────────────────────────────
    # Strip scientific names then tokenise; use prefix matching in LCS so
    # that "beans"/"bean", "wheats"/"wheat" etc. are treated as equivalent.
    return rouge_l_soft_score(ref_val, hyp_val)


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


# ---------------------------------------------------------------------------
# Record-level matching & evaluation
# ---------------------------------------------------------------------------

def find_crop_swap_pairs(cols: List[str]) -> List[Tuple[str, str]]:
    """
    Detect all column pairs that differ only in a trailing ``'1'`` vs ``'2'``.

    Handles both space-separated suffixes (``'Crop species 1'`` /
    ``'Crop species 2'``) and concatenated suffixes (``'N input SC1'`` /
    ``'N input SC2'``).  Swapping **all** pairs simultaneously corresponds to
    relabelling crop-1 as crop-2 and vice-versa across every related field.

    Args:
        cols: Column names to inspect (e.g. the list of shared evaluation fields).

    Returns:
        List of ``(col_1, col_2)`` tuples where ``col_1`` ends in ``'1'`` and
        ``col_2`` is the matching ``'2'`` variant.

    Example:
        >>> cols = ['Crop species 1', 'Crop species 2', 'N input SC1', 'N input SC2', 'Yield unit']
        >>> find_crop_swap_pairs(cols)
        [('Crop species 1', 'Crop species 2'), ('N input SC1', 'N input SC2')]
    """
    col_set = set(cols)
    seen: set = set()
    pairs: List[Tuple[str, str]] = []
    for c in cols:
        if c in seen:
            continue
        for suffix1, suffix2 in [(" 1", " 2"), ("1", "2")]:
            if c.endswith(suffix1):
                partner = c[: -len(suffix1)] + suffix2
                if partner in col_set:
                    pairs.append((c, partner))
                    seen.add(c)
                    seen.add(partner)
                    break
    return pairs


def _pair_scores(
    ext_row: Any,
    gt_row: Any,
    cols: List[str],
) -> Tuple[Dict[str, float], float, bool]:
    """
    Compute per-field similarity between one extracted row and one GT row.

    Also tries swapping all crop-1/crop-2 column pairs in the extracted row
    (in case the model labelled the two crops in reverse order) and returns
    whichever orientation scores higher.

    Args:
        ext_row: Extracted record (pandas Series or dict-like).
        gt_row:  Ground-truth record (pandas Series or dict-like).
        cols:    Field names to score.

    Returns:
        scores  – ``{field: similarity}`` for the winning orientation.
        mean    – Mean similarity across all fields.
        swapped – ``True`` when the swapped crop orientation was chosen.
    """
    def _score_orientation(row_values: Any) -> Tuple[Dict[str, float], float]:
        s = {c: field_similarity_score(row_values.get(c, ""), gt_row.get(c, "")) for c in cols}
        return s, float(np.mean(list(s.values()))) if s else 0.0

    scores_orig, mean_orig = _score_orientation(ext_row)

    swap_pairs = find_crop_swap_pairs(cols)
    if swap_pairs:
        ext_swapped: Dict[str, Any] = dict(ext_row)
        for c1, c2 in swap_pairs:
            ext_swapped[c1] = ext_row.get(c2, "")
            ext_swapped[c2] = ext_row.get(c1, "")
        scores_swap, mean_swap = _score_orientation(ext_swapped)
        if mean_swap > mean_orig:
            return scores_swap, mean_swap, True

    return scores_orig, mean_orig, False


def evaluate_record_pair(
    ext_row: Any,
    gt_row: Any,
    shared_cols: List[str],
) -> Tuple[Dict[str, float], float, bool]:
    """
    Evaluate a single extracted record against a single GT record.

    This is a thin convenience wrapper around :func:`_pair_scores`, exposing the
    same scoring logic used by :func:`match_records_greedy` and
    :func:`evaluate_method_scores` for a single pair instead of whole DataFrames.

    Args:
        ext_row:     Extracted record (pandas Series or dict-like).
        gt_row:      Ground-truth record (pandas Series or dict-like).
        shared_cols: Field names to score.

    Returns:
        field_scores – ``{field: similarity}`` for the best crop orientation.
        mean_score   – Mean similarity across all ``shared_cols``.
        swapped      – ``True`` when the crop-1/crop-2 orientation was swapped.
    """
    return _pair_scores(ext_row, gt_row, shared_cols)


def match_records_greedy(
    ext_df: "pd.DataFrame",
    gt_df: "pd.DataFrame",
    shared_cols: List[str],
) -> List[Tuple[int, int, Dict[str, float], bool]]:
    """
    Greedy one-to-one record matching between extracted and GT rows.

    For every candidate pair the best crop orientation (original vs. swapped)
    is chosen automatically by :func:`_pair_scores`.  Pairs are committed
    greedily in descending order of mean field similarity so that the
    globally highest-scoring pairs are locked in first.

    Args:
        ext_df:      DataFrame of extracted records (one row per record).
        gt_df:       DataFrame of ground-truth records.
        shared_cols: Field names present in both DataFrames to use for scoring.

    Returns:
        List of ``(ext_row_index, gt_row_index, score_map, swapped)`` tuples.
        Unmatched GT rows and unmatched extracted rows are not included;
        callers should account for them in the denominator separately.

    Example:
        >>> matches = match_records_greedy(ext_df, gt_df, shared_fields)
        >>> for ext_i, gt_i, scores, swapped in matches:
        ...     print(ext_i, gt_i, swapped, round(sum(scores.values()) / len(scores), 3))
    """
    candidates = []
    for ext_i, ext_row in ext_df.iterrows():
        for gt_i, gt_row in gt_df.iterrows():
            score_map, mean_score, swapped = _pair_scores(ext_row, gt_row, shared_cols)
            candidates.append((mean_score, ext_i, gt_i, score_map, swapped))

    candidates.sort(key=lambda x: x[0], reverse=True)

    used_ext: set = set()
    used_gt: set = set()
    matches: List[Tuple[int, int, Dict[str, float], bool]] = []
    for _, ext_i, gt_i, score_map, swapped in candidates:
        if ext_i in used_ext or gt_i in used_gt:
            continue
        used_ext.add(ext_i)
        used_gt.add(gt_i)
        matches.append((ext_i, gt_i, score_map, swapped))

    return matches


def evaluate_method_scores(
    ext_df: "pd.DataFrame",
    gt_df: "pd.DataFrame",
    shared_cols: List[str],
    total_records_for_denominator: int,
) -> Tuple["pd.DataFrame", float, Dict[str, float], int]:
    """
    Compute a normalized similarity score for one extraction method.

    Score formula::

        normalized_score = sum(field similarities over matched pairs)
                           / (total_records_for_denominator * n_fields)

    Using ``total_records_for_denominator = len(gt_df)`` penalises methods
    that extract fewer records than the ground truth.

    Args:
        ext_df:                       DataFrame of extracted records.
        gt_df:                        Ground-truth DataFrame.
        shared_cols:                  Fields present in both DataFrames.
        total_records_for_denominator: Denominator for normalisation
                                      (typically ``len(gt_df)``).

    Returns:
        scores_df           – per-match, per-field similarity DataFrame.
        overall_normalized  – scalar summary score in ``[0, 1]``.
        per_field_normalized– ``{'rougeL_<field>': score}`` dict for plotting.
        n_matches           – number of matched record pairs.

    Example:
        >>> scores_df, overall, per_field, n = evaluate_method_scores(
        ...     ext_df, gt_paper, shared_fields, total_records=len(gt_paper))
        >>> print(f'Normalized score: {overall:.3f}  ({n} matched pairs)')
    """
    matches = match_records_greedy(ext_df, gt_df, shared_cols)

    score_rows = [
        {f"rougeL_{c}": score_map.get(c, 0.0) for c in shared_cols}
        for _, _, score_map, _ in matches
    ]
    scores_df = pd.DataFrame(score_rows)

    n_fields  = len(shared_cols)
    n_records = max(int(total_records_for_denominator), 1)
    denom     = n_records * max(n_fields, 1)

    total_sum          = float(scores_df.to_numpy().sum()) if not scores_df.empty else 0.0
    overall_normalized = total_sum / denom

    per_field_normalized = {
        f"rougeL_{c}": (
            float(scores_df[f"rougeL_{c}"].sum()) / n_records
            if f"rougeL_{c}" in scores_df.columns
            else 0.0
        )
        for c in shared_cols
    }

    return scores_df, overall_normalized, per_field_normalized, len(matches)


def parse_field_value_pairs_artifact(raw: Any) -> List[Any]:
    """
    Normalise ``field_value_pairs`` workspace artifacts to a list of pairs.

    Accepts a Python list, or a JSON / markdown-wrapped string as produced by LLMs.
    """
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str):
        return []

    s = raw.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
        s = s[1:-1].strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    try:
        out = json.loads(s)
        return out if isinstance(out, list) else []
    except Exception:
        pass

    m = re.search(r"\[.*\]", s, flags=re.DOTALL)
    if m:
        payload = m.group(0)
        try:
            out = json.loads(payload)
            return out if isinstance(out, list) else []
        except Exception:
            try:
                out = ast.literal_eval(payload)
                return out if isinstance(out, list) else []
            except Exception:
                return []
    return []


def print_field_value_pairs_vs_gt(
    raw_pairs: Any,
    gt_df: "pd.DataFrame",
    shared_cols: List[str],
    show_fields: Optional[List[str]] = None,
) -> None:
    """
    Pretty-print Step-1 ``field_value_pairs`` using the same columns as
    :func:`print_matching_pairs_from_df` (FIELD / EXTRACTED / GT / SCORE).

    Step-1 pairs are not aligned to GT rows. For each GT row and each field,
    **EXTRACTED** is the step-1 value (among all pairs with that field name)
    that maximises :func:`field_similarity_score` against that row's GT cell.
    This makes the layout comparable to record-level prints while reflecting
    the bag-of-evidence nature of the artifact.
    """
    pairs = parse_field_value_pairs_artifact(raw_pairs)
    eval_cols = list(shared_cols)
    if not eval_cols:
        print("  [no shared_cols provided]")
        return

    by_field: Dict[str, List[Any]] = {}
    for p in pairs or []:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        fname, fval = p[0], p[1]
        if not isinstance(fname, str):
            continue
        if fname not in eval_cols:
            continue
        if _is_missing(fval):
            continue
        by_field.setdefault(fname, []).append(fval)

    display_cols = show_fields if show_fields else eval_cols
    n_pairs = len(pairs) if pairs else 0
    n_gt = len(gt_df)

    print(f"  Step-1 pair occurrences : {n_pairs}")
    print(f"  GT rows                 : {n_gt}")
    print(
        "  Note: no row alignment — per GT row, EXTRACTED = best step-1 value "
        "for that field name vs this row's GT cell."
    )
    print()

    col_w = 28
    val_w = 26
    header = f"{'FIELD':{col_w}}  {'EXTRACTED':{val_w}}  {'GT':{val_w}}  SCORE"
    divider = "-" * len(header)

    for gt_i, gt_row in gt_df.iterrows():
        row_scores: List[float] = []
        lines_out: List[Tuple[str, str, str, float]] = []

        for c in display_cols:
            gt_val = gt_row.get(c, "")
            candidates = by_field.get(c, [])

            if not candidates:
                ext_raw: Any = None
                score = field_similarity_score(gt_val, ext_raw)
            else:
                best_v = None
                best_s = -1.0
                for v in candidates:
                    s = field_similarity_score(gt_val, v)
                    if s > best_s:
                        best_s = s
                        best_v = v
                ext_raw = best_v
                score = best_s

            row_scores.append(score)

            if ext_raw is not None and not _is_missing(ext_raw):
                ext_iso = _excel_serial_to_iso(ext_raw)
                ext_str = ext_iso if ext_iso is not None else str(ext_raw)
            else:
                ext_str = "<missing>"

            if not _is_missing(gt_val):
                gt_iso = _excel_serial_to_iso(gt_val)
                gt_str = gt_iso if gt_iso is not None else str(gt_val)
            else:
                gt_str = "<missing>"

            lines_out.append((c, ext_str, gt_str, float(score)))

        mean_score = float(np.mean(row_scores)) if row_scores else 0.0
        print(
            f"  ── GT row {gt_i}  (step-1 best-per-field vs this GT row)  "
            f"mean={mean_score:.3f}"
        )
        print(f"  {header}")
        print(f"  {divider}")
        for c, ext_str, gt_str, score in lines_out:
            print(f"  {c:{col_w}}  {ext_str:{val_w}}  {gt_str:{val_w}}  {score:.3f}")
        print()


def print_matching_pairs_from_df(
    ext_df: "pd.DataFrame",
    gt_df: "pd.DataFrame",
    shared_cols: List[str],
    max_pairs: Optional[int] = None,
    show_fields: Optional[List[str]] = None,
) -> None:
    """
    Pretty-print each greedy-matched (extracted, GT) record pair with per-field scores.

    Same output format as :func:`print_matching_pairs`, but accepts an in-memory
    DataFrame (e.g. MAS workspace artifacts) instead of a CSV path.
    """
    if ext_df.empty:
        print("  [empty extraction]")
        return

    # Use the full shared column list for both matching and display.
    # Missing extracted columns are treated as "<missing>" (score 0.0) instead of being dropped.
    eval_cols = list(shared_cols)
    if not eval_cols:
        print("  [no shared_cols provided]")
        return

    display_cols = show_fields if show_fields else eval_cols
    matches      = match_records_greedy(ext_df, gt_df, eval_cols)
    swap_pairs   = find_crop_swap_pairs(eval_cols)

    n_gt            = len(gt_df)
    n_ext           = len(ext_df)
    n_matched       = len(matches)
    n_swapped       = sum(1 for *_, sw in matches if sw)
    n_gt_unmatched  = n_gt  - n_matched
    n_ext_unmatched = n_ext - n_matched

    print(f"  Extracted rows : {n_ext}")
    print(f"  GT rows        : {n_gt}")
    print(f"  Matched pairs  : {n_matched}  ({n_swapped} used crop-label swap)")
    print(f"  Unmatched GT   : {n_gt_unmatched}  (no extracted record assigned)")
    print(f"  Unmatched ext  : {n_ext_unmatched}  (hallucinated / extra records)")
    print()

    col_w   = 28
    val_w   = 26
    header  = f"{'FIELD':{col_w}}  {'EXTRACTED':{val_w}}  {'GT':{val_w}}  SCORE"
    divider = "-" * len(header)

    pairs_to_show = matches[:max_pairs] if max_pairs else matches
    for rank, (ext_i, gt_i, score_map, swapped) in enumerate(pairs_to_show, 1):
        ext_row    = ext_df.loc[ext_i]
        gt_row     = gt_df.loc[gt_i]
        mean_score = float(np.mean(list(score_map.values()))) if score_map else 0.0
        swap_label = "  [crop labels swapped]" if swapped else ""
        print(f"  ── Pair {rank}  (ext row {ext_i}  ↔  gt row {gt_i})  mean={mean_score:.3f}{swap_label}")
        print(f"  {header}")
        print(f"  {divider}")

        if swapped:
            ext_display: Any = dict(ext_row)
            for c1, c2 in swap_pairs:
                ext_display[c1] = ext_row.get(c2, "")
                ext_display[c2] = ext_row.get(c1, "")
        else:
            ext_display = ext_row

        for c in display_cols:
            ext_val = ext_display.get(c, "")
            gt_val  = gt_row.get(c, "")
            score   = score_map.get(c, field_similarity_score(ext_val, gt_val))
            # Pretty-print values, normalising Excel-serial-style dates when possible
            if not _is_missing(ext_val):
                ext_iso = _excel_serial_to_iso(ext_val)
                ext_str = ext_iso if ext_iso is not None else str(ext_val)
            else:
                ext_str = "<missing>"

            if not _is_missing(gt_val):
                gt_iso = _excel_serial_to_iso(gt_val)
                gt_str = gt_iso if gt_iso is not None else str(gt_val)
            else:
                gt_str = "<missing>"
            print(f"  {c:{col_w}}  {ext_str:{val_w}}  {gt_str:{val_w}}  {score:.3f}")
        print()

    if n_gt_unmatched > 0:
        matched_gt_indices = {gt_i for _, gt_i, *_ in matches}
        unmatched_gt = gt_df.loc[[i for i in gt_df.index if i not in matched_gt_indices]]
        print("  ── Unmatched GT rows (no extracted record assigned) ──")
        for gt_i, gt_row in unmatched_gt.iterrows():
            vals = {
                c: gt_row.get(c, "")
                for c in display_cols
                if not _is_missing(gt_row.get(c, ""))
            }
            print(f"  gt row {gt_i}: { {k: str(v) for k, v in list(vals.items())[:6]} }")
        print()


def print_matching_pairs(
    csv_path: Any,
    gt_df: "pd.DataFrame",
    shared_cols: List[str],
    max_pairs: Optional[int] = None,
    show_fields: Optional[List[str]] = None,
) -> None:
    """
    Load a method's output CSV and pretty-print each matched GT pair
    side-by-side with per-field similarity scores.

    For each matched pair the function shows:

    * The extracted value, the GT value, and the similarity score for every
      display field.
    * A ``[crop labels swapped]`` label when the swap orientation scored higher.
    * A summary of unmatched GT rows (records the method failed to extract).

    Args:
        csv_path:    Path to the extracted-records CSV file.
        gt_df:       Ground-truth DataFrame (e.g. ``gt_paper``).
        shared_cols: Fields to use for matching (e.g. ``shared_fields``).
        max_pairs:   Maximum number of matched pairs to print (``None`` = all).
        show_fields: Subset of fields to display per pair.
                     Defaults to all fields in ``shared_cols``.

    Example:
        >>> print_matching_pairs(
        ...     csv_path=path_mas,
        ...     gt_df=gt_paper,
        ...     shared_cols=shared_fields,
        ...     show_fields=['Crop species 1', 'Crop species 2',
        ...                  'unified yield ic 1', 'unified yield ic 2'],
        ... )
    """
    if csv_path is None or not os.path.exists(str(csv_path)):
        print("  [no CSV found]")
        return

    ext_df = pd.read_csv(csv_path)
    if ext_df.empty:
        print("  [empty CSV]")
        return

    print_matching_pairs_from_df(
        ext_df,
        gt_df,
        shared_cols,
        max_pairs=max_pairs,
        show_fields=show_fields,
    )


__all__ = [
    "load_ground_truth",
    "build_study_paper_mapping",
    "get_paper_path_for_study",
    "highlight_numbers_and_tables",
    "build_extraction_context",
    "rouge_l_soft_score",
    "field_similarity_score",
    "find_crop_swap_pairs",
    "evaluate_record_pair",
    "match_records_greedy",
    "evaluate_method_scores",
    "print_matching_pairs",
    "print_matching_pairs_from_df",
    "parse_field_value_pairs_artifact",
    "print_field_value_pairs_vs_gt",
    "_is_missing",
]
