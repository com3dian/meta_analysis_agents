"""
Prepare GenAI extraction rows for LER–TND reproduction.

Decision rule for each GenAI row:

    Usable reported LER?
        yes → use reported LER
        no  → usable four yield values with valid units?
                  yes → normalize yields and calculate LER
                  no  → exclude row

Reported LER always takes precedence. Yield-derived LER is still computed
when possible as an audit check and never silently replaces a usable reported
value.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .eval_utils import (
    NUMERIC_RELATIVE_TOLERANCE,
    _excel_serial_to_iso,
    _is_missing,
    _parse_absolute_iso_date_from_text,
    _parse_relative_days_value,
    _try_parse_number,
)
from .units import Quantity, Unit, canonicalize_yield_unit, sanitize_unit_string

# ---------------------------------------------------------------------------
# Field names
# ---------------------------------------------------------------------------

YIELD_FIELDS: Tuple[str, str, str, str] = (
    "unified yield sc 1",
    "unified yield sc 2",
    "unified yield ic 1",
    "unified yield ic 2",
)
YIELD_UNIT_FIELD = "Yield unit"
LER_FIELD = "LER"
TND_FIELD = "TND"
DATE_FIELDS: Tuple[str, str, str, str] = (
    "Sowing date 1",
    "Sowing date 2",
    "Harvest date 1",
    "Harvest date 2",
)

CANONICAL_YIELD_UNIT = "t/ha"
LER_UNUSUAL_THRESHOLD = 3.0
TND_MIN = 0.0
TND_MAX = 1.0

# Range / multi-value patterns that make a reported LER unusable.
_RANGE_SEPARATOR_RE = re.compile(
    r"(?:\.{2,}|…|–|—|−|-/|to\b|and\b)",
    re.IGNORECASE,
)
_NUMBER_TOKEN_RE = re.compile(
    r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?"
)
_INLINE_UNIT_RE = re.compile(
    r"^\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*"
    r"(.+?)\s*$"
)
_WEEKS_LATER_RE = re.compile(
    r"(\d{1,3})(?:\s*(?:-|to|and|–|—)\s*(\d{1,3}))?\s*weeks?\s+later",
    re.IGNORECASE,
)
_DAYS_LATER_RE = re.compile(
    r"(\d{1,3})(?:\s*(?:-|to|and|–|—)\s*(\d{1,3}))?\s*days?\s+later",
    re.IGNORECASE,
)


@dataclass
class PreparedLERRow:
    """Auditable LER / yield / TND preparation for one GenAI record."""

    LER_reported_raw: Any = None
    LER_reported_numeric: Optional[float] = None
    yield_sc1_raw: Any = None
    yield_sc2_raw: Any = None
    yield_ic1_raw: Any = None
    yield_ic2_raw: Any = None
    yield_unit_raw: Any = None
    yield_unit_canonical: Optional[str] = None
    yield_sc1_t_ha: Optional[float] = None
    yield_sc2_t_ha: Optional[float] = None
    yield_ic1_t_ha: Optional[float] = None
    yield_ic2_t_ha: Optional[float] = None
    LER_from_yields: Optional[float] = None
    LER_final: Optional[float] = None
    LER_source: str = "excluded"
    exclusion_reason: Optional[str] = None
    LER_unusual_flag: bool = False
    LER_agreement: str = "not_checkable"
    LER_relative_difference: Optional[float] = None
    yield_unit_conflict: Optional[str] = None
    TND_reported_raw: Any = None
    TND_reported_numeric: Optional[float] = None
    TND_from_dates: Optional[float] = None
    TND_final: Optional[float] = None
    TND_source: str = "excluded"
    TND_exclusion_reason: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _finite_positive(value: Optional[float]) -> bool:
    return value is not None and math.isfinite(value) and value > 0.0


def _finite_nonnegative(value: Optional[float]) -> bool:
    return value is not None and math.isfinite(value) and value >= 0.0


def parse_single_numeric(value: Any) -> Tuple[Optional[float], Optional[str]]:
    """
    Parse a cell that must contain exactly one usable number.

    Returns ``(number, exclusion_reason)``. Reason is set when unusable.
    """
    if _is_missing(value):
        return None, "missing_value"

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        num = float(value)
        if not math.isfinite(num):
            return None, "non_finite_numeric"
        return num, None

    text = str(value).strip()
    if not text:
        return None, "missing_value"

    # Pure number (with optional thousands commas).
    pure = _try_parse_number(text)
    if pure is not None:
        if not math.isfinite(pure):
            return None, "non_finite_numeric"
        return pure, None

    # Reject explicit ranges / multi-value text before extracting numbers.
    numbers = [float(m.group(0).replace(",", "")) for m in _NUMBER_TOKEN_RE.finditer(text)]
    if len(numbers) == 0:
        return None, "unparseable_numeric"
    if len(numbers) > 1 or _RANGE_SEPARATOR_RE.search(text):
        return None, "ambiguous_numeric_range"

    num = numbers[0]
    if not math.isfinite(num):
        return None, "non_finite_numeric"
    return num, None


def parse_usable_reported_ler(value: Any) -> Tuple[Optional[float], Optional[str]]:
    """Return usable reported LER (> 0, finite, single number) or a reason."""
    num, reason = parse_single_numeric(value)
    if reason is not None:
        if reason == "missing_value":
            return None, "missing_reported_ler"
        if reason == "ambiguous_numeric_range":
            return None, "ambiguous_numeric_range"
        if reason == "unparseable_numeric":
            return None, "unparseable_reported_ler"
        return None, reason
    assert num is not None
    if num <= 0.0:
        return None, "non_positive_reported_ler"
    return num, None


def parse_usable_reported_tnd(value: Any) -> Tuple[Optional[float], Optional[str]]:
    """Return usable reported TND in ``[0, 1]`` or a reason."""
    num, reason = parse_single_numeric(value)
    if reason is not None:
        if reason == "missing_value":
            return None, "missing_reported_tnd"
        if reason == "ambiguous_numeric_range":
            return None, "ambiguous_tnd_range"
        return None, "unparseable_reported_tnd"
    assert num is not None
    if num < TND_MIN or num > TND_MAX:
        return None, "tnd_out_of_range"
    return num, None


def _split_value_and_inline_unit(raw: Any) -> Tuple[Any, Optional[str]]:
    """If a yield cell embeds a unit (e.g. ``8.5 t/ha``), split it out."""
    if _is_missing(raw) or isinstance(raw, (int, float)):
        return raw, None
    text = str(raw).strip()
    if not text:
        return raw, None
    if _try_parse_number(text) is not None:
        return text, None
    m = _INLINE_UNIT_RE.match(text)
    if not m:
        return raw, None
    unit_part = m.group(2).strip()
    # Require the trailing part to look like a unit, not prose.
    if not re.search(r"[A-Za-z]", unit_part):
        return raw, None
    if _RANGE_SEPARATOR_RE.search(text) and len(_NUMBER_TOKEN_RE.findall(text)) > 1:
        return raw, None
    return m.group(1), unit_part


def _is_mass_per_area_yield_unit(unit: Unit) -> bool:
    d = unit.dimension
    return (
        d.mass_exp == 1
        and d.length_exp == -2
        and d.count_exp == 0
        and d.substance in (None, "DM")
    )


def canonicalize_genai_yield_unit(unit_raw: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Canonicalize a GenAI yield unit for conversion to ``t/ha``.

    Uses ``for_gt=False``. Returns ``(canonical_symbol, error_reason)``.
    """
    if _is_missing(unit_raw):
        return None, "missing_yield_unit"
    cleaned = sanitize_unit_string(str(unit_raw))
    if not cleaned:
        return None, "missing_yield_unit"
    canon = canonicalize_yield_unit(cleaned, for_gt=False)
    try:
        unit = Unit(canon)
    except (ValueError, TypeError):
        return None, "incompatible_yield_unit"
    if not _is_mass_per_area_yield_unit(unit):
        return None, "incompatible_yield_unit"
    # Drop DM substance so conversion to plain t/ha succeeds.
    if unit.dimension.substance == "DM":
        canon = "g/m2"
        unit = Unit(canon)
    return canon, None


def convert_yield_to_t_ha(
    value_raw: Any,
    unit_raw: Any,
) -> Tuple[Optional[float], Optional[str], Optional[str], Optional[str]]:
    """
    Convert one yield cell to ``t/ha``.

    Returns ``(t_ha, canonical_unit, inline_unit_or_None, error_reason)``.
    """
    value_part, inline_unit = _split_value_and_inline_unit(value_raw)
    effective_unit = unit_raw

    if inline_unit is not None:
        if _is_missing(unit_raw):
            effective_unit = inline_unit
        else:
            try:
                u_row = Unit(canonicalize_yield_unit(unit_raw, for_gt=False))
                u_inline = Unit(canonicalize_yield_unit(inline_unit, for_gt=False))
                if not u_row.is_compatible(u_inline) or abs(
                    u_row.conversion_factor_to(u_inline) - 1.0
                ) > 1e-9:
                    return None, None, inline_unit, "incompatible_yield_unit"
            except (ValueError, TypeError):
                return None, None, inline_unit, "incompatible_yield_unit"
            effective_unit = unit_raw

    mag, mag_reason = parse_single_numeric(value_part)
    if mag_reason is not None:
        if mag_reason == "missing_value":
            return None, None, inline_unit, "missing_yield_value"
        if mag_reason == "ambiguous_numeric_range":
            return None, None, inline_unit, "ambiguous_numeric_range"
        return None, None, inline_unit, "unparseable_yield_value"

    canon, unit_reason = canonicalize_genai_yield_unit(effective_unit)
    if unit_reason is not None:
        return None, None, inline_unit, unit_reason

    assert mag is not None and canon is not None
    try:
        qty = Quantity(mag, Unit(canon)).to(CANONICAL_YIELD_UNIT)
    except (ValueError, TypeError, ZeroDivisionError):
        return None, canon, inline_unit, "incompatible_yield_unit"
    if not math.isfinite(qty.magnitude):
        return None, canon, inline_unit, "non_finite_yield_value"
    return float(qty.magnitude), canon, inline_unit, None


def calculate_ler_from_yields(
    sc1: Optional[float],
    sc2: Optional[float],
    ic1: Optional[float],
    ic2: Optional[float],
) -> Tuple[Optional[float], Optional[str]]:
    """Total LER from four ``t/ha`` yields. Never partial."""
    if sc1 is None or sc2 is None or ic1 is None or ic2 is None:
        missing = []
        if sc1 is None:
            missing.append("missing_sole_yield_1")
        if sc2 is None:
            missing.append("missing_sole_yield_2")
        if ic1 is None:
            missing.append("missing_intercrop_yield_1")
        if ic2 is None:
            missing.append("missing_intercrop_yield_2")
        return None, missing[0] if len(missing) == 1 else "missing_ler_and_yields"

    if sc1 <= 0.0 or sc2 <= 0.0:
        return None, "zero_sole_crop_yield"
    if not all(math.isfinite(v) for v in (sc1, sc2, ic1, ic2)):
        return None, "non_finite_yield_value"
    if ic1 < 0.0 or ic2 < 0.0:
        return None, "negative_intercrop_yield"

    ler = (ic1 / sc1) + (ic2 / sc2)
    if not math.isfinite(ler) or ler <= 0.0:
        return None, "invalid_calculated_ler"
    return float(ler), None


def relative_difference(a: float, b: float) -> Optional[float]:
    """Signed relative difference ``(a - b) / |b|``; None if undefined."""
    if b == 0.0 or not math.isfinite(a) or not math.isfinite(b):
        return None
    return (a - b) / abs(b)


def ler_agreement_status(
    reported: Optional[float],
    from_yields: Optional[float],
    *,
    tol: float = float(NUMERIC_RELATIVE_TOLERANCE),
) -> Tuple[str, Optional[float]]:
    if reported is None or from_yields is None:
        return "not_checkable", None
    rel = relative_difference(reported, from_yields)
    if rel is None:
        return "not_checkable", None
    if abs(rel) <= tol:
        return "agree", rel
    return "disagree", rel


# ---------------------------------------------------------------------------
# TND from dates
# ---------------------------------------------------------------------------


def _iso_to_ordinal(iso: str) -> Optional[int]:
    try:
        return date.fromisoformat(iso).toordinal()
    except ValueError:
        return None


def _parse_weeks_later(value: Any) -> Optional[float]:
    if _is_missing(value):
        return None
    text = str(value).strip()
    m = _WEEKS_LATER_RE.search(text)
    if not m:
        return None
    a = float(m.group(1))
    b = float(m.group(2)) if m.group(2) is not None else a
    return 7.0 * (a + b) / 2.0


def _parse_days_later(value: Any) -> Optional[float]:
    if _is_missing(value):
        return None
    text = str(value).strip()
    m = _DAYS_LATER_RE.search(text)
    if not m:
        return None
    a = float(m.group(1))
    b = float(m.group(2)) if m.group(2) is not None else a
    return (a + b) / 2.0


def parse_date_to_day_number(value: Any, *, origin: Optional[int] = None) -> Optional[float]:
    """
    Parse a sowing/harvest expression into a comparable day number.

    Supports Excel serials, absolute calendar dates, and relative
    ``N days/weeks later`` expressions (requires ``origin`` = sowing day).
    """
    if _is_missing(value):
        return None

    if isinstance(value, datetime):
        return float(value.date().toordinal())
    if isinstance(value, date):
        return float(value.toordinal())

    # Absolute first (includes Excel serial via shared helper).
    iso = _parse_absolute_iso_date_from_text(value)
    if iso is not None:
        ord_ = _iso_to_ordinal(iso)
        return float(ord_) if ord_ is not None else None

    # Excel serial fallback for values outside the absolute helper's text paths.
    excel_iso = _excel_serial_to_iso(value)
    if excel_iso is not None:
        ord_ = _iso_to_ordinal(excel_iso)
        return float(ord_) if ord_ is not None else None

    if origin is not None:
        weeks = _parse_weeks_later(value)
        if weeks is not None:
            return float(origin) + weeks
        days = _parse_days_later(value)
        if days is not None:
            return float(origin) + days
        rel = _parse_relative_days_value(value)
        if rel is not None:
            lo, hi = rel
            return float(origin) + (lo + hi) / 2.0

    return None


def calculate_tnd(
    sowing1: Any,
    sowing2: Any,
    harvest1: Any,
    harvest2: Any,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Temporal niche differentiation from four dates.

    Matches the Yu et al. spreadsheet definition on absolute Excel dates:

        span = max(h1, h2) - min(s1, s2) + 1
        overlap = max(0, min(h1, h2) - max(s1, s2) + 1)
        TND = 1 - overlap / span

    Both crops must have harvest strictly after sowing.
    """
    s1 = parse_date_to_day_number(sowing1)
    s2 = parse_date_to_day_number(sowing2)
    if s1 is None or s2 is None:
        return None, "unparseable_sowing_date"

    h1 = parse_date_to_day_number(harvest1, origin=int(round(s1)))
    h2 = parse_date_to_day_number(harvest2, origin=int(round(s2)))
    if h1 is None or h2 is None:
        return None, "unparseable_harvest_date"

    if h1 <= s1 or h2 <= s2:
        return None, "invalid_growth_period"

    span = max(h1, h2) - min(s1, s2) + 1.0
    if span <= 0:
        return None, "invalid_growth_period"
    overlap = max(0.0, min(h1, h2) - max(s1, s2) + 1.0)
    tnd = 1.0 - (overlap / span)
    if not math.isfinite(tnd):
        return None, "invalid_calculated_tnd"
    # Clamp tiny numeric noise into [0, 1].
    if tnd < -1e-9 or tnd > 1.0 + 1e-9:
        return None, "tnd_out_of_range"
    tnd = min(1.0, max(0.0, tnd))
    return float(tnd), None


# ---------------------------------------------------------------------------
# Row / table preparation
# ---------------------------------------------------------------------------


def prepare_ler_row(row: Mapping[str, Any]) -> PreparedLERRow:
    """Apply the LER / yield / TND decision rules to one GenAI record."""
    out = PreparedLERRow(
        LER_reported_raw=row.get(LER_FIELD),
        yield_sc1_raw=row.get(YIELD_FIELDS[0]),
        yield_sc2_raw=row.get(YIELD_FIELDS[1]),
        yield_ic1_raw=row.get(YIELD_FIELDS[2]),
        yield_ic2_raw=row.get(YIELD_FIELDS[3]),
        yield_unit_raw=row.get(YIELD_UNIT_FIELD),
        TND_reported_raw=row.get(TND_FIELD),
    )

    # --- Yields (always attempt; used for calculation and audit) ---
    yield_errors: List[str] = []
    converted: List[Optional[float]] = []
    canon_unit: Optional[str] = None
    for raw in (
        out.yield_sc1_raw,
        out.yield_sc2_raw,
        out.yield_ic1_raw,
        out.yield_ic2_raw,
    ):
        t_ha, canon, inline, err = convert_yield_to_t_ha(raw, out.yield_unit_raw)
        if canon and canon_unit is None:
            canon_unit = canon
        if inline and not _is_missing(out.yield_unit_raw) and err == "incompatible_yield_unit":
            out.yield_unit_conflict = f"{inline!r}_vs_{out.yield_unit_raw!r}"
        if err:
            yield_errors.append(err)
            converted.append(None)
        else:
            converted.append(t_ha)

    out.yield_unit_canonical = canon_unit
    out.yield_sc1_t_ha, out.yield_sc2_t_ha, out.yield_ic1_t_ha, out.yield_ic2_t_ha = (
        converted
    )

    ler_yield, yield_ler_reason = calculate_ler_from_yields(*converted)
    out.LER_from_yields = ler_yield

    # --- Reported LER precedence ---
    reported, reported_reason = parse_usable_reported_ler(out.LER_reported_raw)
    out.LER_reported_numeric = reported

    if reported is not None:
        out.LER_final = reported
        out.LER_source = "reported"
        out.exclusion_reason = None
        out.LER_unusual_flag = reported > LER_UNUSUAL_THRESHOLD
    elif ler_yield is not None:
        out.LER_final = ler_yield
        out.LER_source = "calculated_from_yields"
        out.exclusion_reason = None
        out.LER_unusual_flag = ler_yield > LER_UNUSUAL_THRESHOLD
    else:
        out.LER_final = None
        out.LER_source = "excluded"
        candidates: List[str] = []
        if reported_reason:
            candidates.append(reported_reason)
        candidates.extend(yield_errors)
        if yield_ler_reason:
            candidates.append(yield_ler_reason)
        priority = [
            "ambiguous_numeric_range",
            "incompatible_yield_unit",
            "missing_yield_unit",
            "unparseable_yield_value",
            "unparseable_reported_ler",
            "non_positive_reported_ler",
            "zero_sole_crop_yield",
            "negative_intercrop_yield",
            "missing_sole_yield_1",
            "missing_sole_yield_2",
            "missing_intercrop_yield_1",
            "missing_intercrop_yield_2",
            "missing_yield_value",
            "missing_reported_ler",
            "missing_ler_and_yields",
        ]
        chosen = None
        for p in priority:
            if p in candidates:
                chosen = p
                break
        out.exclusion_reason = chosen or (candidates[0] if candidates else "missing_ler_and_yields")

    agree, rel = ler_agreement_status(out.LER_reported_numeric, out.LER_from_yields)
    out.LER_agreement = agree
    out.LER_relative_difference = rel

    # --- TND hierarchy ---
    tnd_rep, tnd_rep_reason = parse_usable_reported_tnd(out.TND_reported_raw)
    out.TND_reported_numeric = tnd_rep
    tnd_dates, tnd_date_reason = calculate_tnd(
        row.get(DATE_FIELDS[0]),
        row.get(DATE_FIELDS[1]),
        row.get(DATE_FIELDS[2]),
        row.get(DATE_FIELDS[3]),
    )
    out.TND_from_dates = tnd_dates

    if tnd_rep is not None:
        out.TND_final = tnd_rep
        out.TND_source = "reported"
        out.TND_exclusion_reason = None
    elif tnd_dates is not None:
        out.TND_final = tnd_dates
        out.TND_source = "calculated_from_dates"
        out.TND_exclusion_reason = None
    else:
        out.TND_final = None
        out.TND_source = "excluded"
        out.TND_exclusion_reason = tnd_rep_reason or tnd_date_reason or "missing_tnd"

    return out


def prepare_ler_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    extra_fields: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Prepare many GenAI rows, preserving original columns plus audit fields.

    Original columns are never overwritten; prepared fields use distinct names.
    """
    keep = list(extra_fields) if extra_fields is not None else None
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        prepared = prepare_ler_row(row)
        merged: Dict[str, Any] = {}
        if keep is None:
            merged.update(dict(row))
        else:
            for key in keep:
                if key in row:
                    merged[key] = row[key]
        merged["row_index"] = idx
        merged.update(prepared.as_dict())
        out.append(merged)
    return out


def exclusion_summary(
    records: Iterable[Mapping[str, Any]],
    *,
    model_tag: Optional[str] = None,
    method: Optional[str] = None,
) -> Dict[str, Any]:
    """Count LER / TND sources and exclusion reasons."""
    n = 0
    ler_sources: Dict[str, int] = {}
    tnd_sources: Dict[str, int] = {}
    ler_excl: Dict[str, int] = {}
    tnd_excl: Dict[str, int] = {}
    agree_counts: Dict[str, int] = {}
    unusual = 0
    for rec in records:
        n += 1
        ls = str(rec.get("LER_source", "excluded"))
        ler_sources[ls] = ler_sources.get(ls, 0) + 1
        ts = str(rec.get("TND_source", "excluded"))
        tnd_sources[ts] = tnd_sources.get(ts, 0) + 1
        if ls == "excluded":
            reason = str(rec.get("exclusion_reason") or "unknown")
            ler_excl[reason] = ler_excl.get(reason, 0) + 1
        if ts == "excluded":
            reason = str(rec.get("TND_exclusion_reason") or "unknown")
            tnd_excl[reason] = tnd_excl.get(reason, 0) + 1
        ag = str(rec.get("LER_agreement") or "not_checkable")
        agree_counts[ag] = agree_counts.get(ag, 0) + 1
        if rec.get("LER_unusual_flag"):
            unusual += 1
    return {
        "model_tag": model_tag,
        "method": method,
        "n_rows": n,
        "LER_source_counts": ler_sources,
        "TND_source_counts": tnd_sources,
        "LER_exclusion_reasons": ler_excl,
        "TND_exclusion_reasons": tnd_excl,
        "LER_agreement_counts": agree_counts,
        "n_unusual_ler": unusual,
    }
