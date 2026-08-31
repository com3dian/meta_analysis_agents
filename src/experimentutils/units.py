"""
Unit registry, schema field pairs, and dimensional arithmetic for wopke_100.

``UnitStorage`` lists unit strings observed in annotations and predictions.
``ValueUnitGroup`` maps numeric value columns to their shared unit column.
``Unit`` / ``Quantity`` support scalar multiplication and conversion between
compatible units (e.g. ``50 * unit("kg ha-1") == 50000 * unit("g ha-1")``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Final, Optional, Tuple, Union

Number = Union[int, float, Fraction]


# ---------------------------------------------------------------------------
# Registry of observed unit strings
# ---------------------------------------------------------------------------


class UnitStorage:
    """
    Registry of unit strings observed in ground-truth annotations and
    method predictions for the wopke_100 schema.
    """

    GT_ANNOTATION_COLUMNS: Final[dict[str, str]] = {
        "general": "Unit",
        "density": "Unit of density",
        "n_fertilizer": "Unit.1",
        "p_fertilizer": "Unit.2",
        "k_fertilizer": "Unit.3",
        "yield": "Yield unit",
    }

    PREDICTION_FIELDS: Final[dict[str, str]] = {
        "n_fertilizer": "N Unit",
        "p_fertilizer": "P Unit",
        "k_fertilizer": "K Unit",
        "yield": "Yield unit",
    }

    GT_TO_PREDICTION_FIELD: Final[dict[str, str]] = {
        "Unit.1": "N Unit",
        "Unit.2": "P Unit",
        "Unit.3": "K Unit",
        "Yield unit": "Yield unit",
    }

    GENERAL: Final[list[str]] = ["m"]

    DENSITY: Final[list[str]] = [
        "grains/m2",
        "kg seeds/ha",
        "kg/ha",
        "plants or g/m2",
        "plants/ha",
        "plants/m2",
        "seedlings/m2",
        "seeds or plants/m2",
        "seeds/5 m row",
        "seeds/m2",
    ]

    N: Final[list[str]] = [
        "g N m-2",
        "kg ha-1",
        "kg ha−1",
        "kg N ha-1",
        "kg N ha⁻¹",
        "kg N ha−1",
        "kg N/ha",
        "Kg N/ha",
        "kg/ha",
    ]

    P: Final[list[str]] = [
        "kg diammonium phosphate/ha",
        "kg ha-1",
        "kg ha−1",
        "kg P ha-1",
        "kg P ha⁻¹",
        "kg P ha−1",
        "kg P/ha",
        "kg P2O5 /ha",
        "kg P2O5 ha-1",
        "kg P2O5 ha−1",
        "kg P2O5/ha",
        "kg P₂O₅ ha⁻¹",
        "kg/ha",
        "mg/kg dried soil",
    ]

    K: Final[list[str]] = [
        "kg ha-1",
        "kg K ha-1",
        "kg K ha−1",
        "Kg K/ha",
        "kg K/ha",
        "kg K2O ha-1",
        "kg K2O ha−1",
        "kg K2O/ha",
        "kg K₂O ha⁻¹",
        "kg/ha",
        "mg/kg dried soil",
    ]

    YIELD: Final[list[str]] = [
        "g DM m-2",
        "g m-2",
        "g/m2",
        "kg ha-1",
        "kg ha⁻¹",
        "kg ha−1",
        "kg/ha",
        "kg/m2",
        "Mg ha-1",
        "Mg ha⁻¹",
        "Mg ha−1",
        "Mg/ha",
        "q/ha",
        "t ha-1",
        "t ha⁻¹",
        "t ha−1",
        "t/ha",
    ]

    ALL: Final[list[str]] = [
        "g DM m-2",
        "g m-2",
        "g N m-2",
        "g/m2",
        "grains/m2",
        "kg diammonium phosphate/ha",
        "kg ha-1",
        "kg ha⁻¹",
        "kg ha−1",
        "kg K ha-1",
        "kg K ha−1",
        "kg K/ha",
        "Kg K/ha",
        "kg K2O ha-1",
        "kg K2O ha−1",
        "kg K2O/ha",
        "kg K₂O ha⁻¹",
        "kg N ha-1",
        "kg N ha⁻¹",
        "kg N ha−1",
        "Kg N/ha",
        "kg N/ha",
        "kg P ha-1",
        "kg P ha⁻¹",
        "kg P ha−1",
        "kg P/ha",
        "kg P2O5 /ha",
        "kg P2O5 ha-1",
        "kg P2O5 ha−1",
        "kg P2O5/ha",
        "kg P₂O₅ ha⁻¹",
        "kg seeds/ha",
        "kg/ha",
        "kg/m2",
        "m",
        "Mg ha-1",
        "Mg ha⁻¹",
        "Mg ha−1",
        "Mg/ha",
        "mg/kg dried soil",
        "plants or g/m2",
        "plants/ha",
        "plants/m2",
        "q/ha",
        "seedlings/m2",
        "seeds or plants/m2",
        "seeds/5 m row",
        "seeds/m2",
        "t ha-1",
        "t ha⁻¹",
        "t ha−1",
        "t/ha",
    ]

    @classmethod
    def known(cls) -> frozenset[str]:
        return frozenset(cls.ALL)

    @classmethod
    def for_field(cls, field_name: str) -> list[str]:
        pred = cls.GT_TO_PREDICTION_FIELD.get(field_name, field_name)
        if pred == "N Unit":
            return list(cls.N)
        if pred == "P Unit":
            return list(cls.P)
        if pred == "K Unit":
            return list(cls.K)
        if pred == "Yield unit":
            return list(cls.YIELD)
        if field_name == cls.GT_ANNOTATION_COLUMNS["density"]:
            return list(cls.DENSITY)
        if field_name == cls.GT_ANNOTATION_COLUMNS["general"]:
            return list(cls.GENERAL)
        return list(cls.ALL)


# ---------------------------------------------------------------------------
# Value + unit field groups (wopke_100 / 42-field schema)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValueUnitGroup:
    """One unit column shared by multiple numeric value columns on the same row."""

    unit_field: str
    value_fields: Tuple[str, ...]
    description: str = ""


# Prediction CSV / 42-field schema (METADATA_STANDARDS["wopke_100"]).
WOPKE_100_VALUE_UNIT_GROUPS: Final[Tuple[ValueUnitGroup, ...]] = (
    ValueUnitGroup(
        unit_field="N Unit",
        value_fields=(
            "N input SC1",
            "N input SC2",
            "N input IC1",
            "N input IC2",
            "N total in IC",
        ),
        description="Nitrogen fertilizer amounts (sole crop + intercrop).",
    ),
    ValueUnitGroup(
        unit_field="P Unit",
        value_fields=(
            "P input SC1",
            "P input SC2",
            "P input IC1",
            "P input IC2",
            "P total in IC",
        ),
        description="Phosphorus fertilizer amounts.",
    ),
    ValueUnitGroup(
        unit_field="K Unit",
        value_fields=(
            "K input SC1",
            "K input SC2",
            "K input IC1",
            "K input IC2",
            "K total in IC",
        ),
        description="Potassium fertilizer amounts.",
    ),
    ValueUnitGroup(
        unit_field="Yield unit",
        value_fields=(
            "unified yield sc 1",
            "unified yield sc 2",
            "unified yield ic 1",
            "unified yield ic 2",
        ),
        description="Standardised crop yields (sole crop + intercrop per species).",
    ),
)

# Ground-truth spreadsheet uses deduplicated headers (see UnitStorage.GT_ANNOTATION_COLUMNS).
WOPKE_100_GT_VALUE_UNIT_GROUPS: Final[Tuple[ValueUnitGroup, ...]] = (
    ValueUnitGroup(
        unit_field="Unit.1",
        value_fields=(
            "N input SC1",
            "N input SC2",
            "N input IC1",
            "N input IC2",
            "N total in IC",
        ),
        description="GT column Unit.1 → same role as N Unit in predictions.",
    ),
    ValueUnitGroup(
        unit_field="Unit.2",
        value_fields=(
            "P input SC1",
            "P input SC2",
            "P input IC1",
            "P input IC2",
            "P total in IC",
        ),
        description="GT column Unit.2 → same role as P Unit.",
    ),
    ValueUnitGroup(
        unit_field="Unit.3",
        value_fields=(
            "K input SC1",
            "K input SC2",
            "K input IC1",
            "K input IC2",
            "K total in IC",
        ),
        description="GT column Unit.3 → same role as K Unit.",
    ),
    ValueUnitGroup(
        unit_field="Unit of density",
        value_fields=(
            "Density ic 1",
            "Density ic 2",
            "Density sc 1",
            "Density sc 2",
        ),
        description="Plant density (GT only; not in 42-field prediction CSVs).",
    ),
    ValueUnitGroup(
        unit_field="Yield unit",
        value_fields=(
            "unified yield sc 1",
            "unified yield sc 2",
            "unified yield ic 1",
            "unified yield ic 2",
        ),
        description="Yield unit shared by all unified yield fields.",
    ),
)

# Flat lookup: value_field → unit_field (prediction schema).
VALUE_TO_UNIT_FIELD: Final[Dict[str, str]] = {
    value: group.unit_field
    for group in WOPKE_100_VALUE_UNIT_GROUPS
    for value in group.value_fields
}

# Unit columns are record-level; they are NOT swapped when crop 1/2 labels swap.
UNIT_FIELDS: Final[Tuple[str, ...]] = tuple(g.unit_field for g in WOPKE_100_VALUE_UNIT_GROUPS)


def value_unit_groups(*, gt_spreadsheet: bool = False) -> Tuple[ValueUnitGroup, ...]:
    """Return value+unit groups for the prediction schema or GT spreadsheet headers."""
    return WOPKE_100_GT_VALUE_UNIT_GROUPS if gt_spreadsheet else WOPKE_100_VALUE_UNIT_GROUPS


def unit_field_for_value(value_field: str, *, gt_spreadsheet: bool = False) -> Optional[str]:
    """Return the unit column name for a numeric value field, or None."""
    if gt_spreadsheet:
        for group in WOPKE_100_GT_VALUE_UNIT_GROUPS:
            if value_field in group.value_fields:
                return group.unit_field
        return None
    return VALUE_TO_UNIT_FIELD.get(value_field)


# ---------------------------------------------------------------------------
# Dimensional unit parsing and conversion
# ---------------------------------------------------------------------------

_MASS_TO_G: dict[str, float] = {
    "g": 1.0,
    "kg": 1_000.0,
    "mg": 0.001,
    "Mg": 1_000_000.0,  # megagram (tonne)
    "t": 1_000_000.0,
    "q": 100_000.0,  # quintal = 100 kg
}

_LENGTH_TO_M: dict[str, float] = {
    "m": 1.0,
}

_AREA_TO_M2: dict[str, float] = {
    "m2": 1.0,
    "m-2": 1.0,
    "ha": 10_000.0,
}

# Optional nutrient / dry-matter label on a mass unit (must match to convert).
_SUBSTANCE_TAGS = frozenset(
    {"N", "P", "P2O5", "K", "K2O", "DM", "diammonium", "phosphate", "dried", "soil"}
)

_COUNT_BASES = frozenset(
    {
        "plant",
        "plants",
        "seed",
        "seeds",
        "seedling",
        "seedlings",
        "grain",
        "grains",
    }
)

_UNICODE_MINUS = str.maketrans("−⁻", "--")


def _normalize_unit_text(symbol: str) -> str:
    s = symbol.strip().translate(_UNICODE_MINUS)
    s = re.sub(r"(?<![a-z])ha[-⁻−\s]*¹?", "ha-1", s, flags=re.IGNORECASE)
    s = re.sub(r"m2[-⁻−\s]*¹?", "m2", s, flags=re.IGNORECASE)
    s = re.sub(r"[¹²³]", "", s)
    s = s.replace("₂", "2").replace("₅", "5")
    # Megagram (Mg) must not collapse to milligram (mg) when lowercased.
    s = re.sub(r"\bMg\b", "MEGAGRAM", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower().replace("megagram", "Mg")


@dataclass(frozen=True)
class _Dimension:
    """Internal SI-like dimensions: grams^a · metres^b · counts^c + substance tag."""

    mass_exp: Fraction = Fraction(0)
    length_exp: Fraction = Fraction(0)
    count_exp: Fraction = Fraction(0)
    substance: Optional[str] = None  # e.g. "N", "P2O5", "DM"

    def is_compatible(self, other: _Dimension) -> bool:
        return (
            self.mass_exp == other.mass_exp
            and self.length_exp == other.length_exp
            and self.count_exp == other.count_exp
            and self.substance == other.substance
        )


def _parse_unit_symbol(symbol: str) -> tuple[_Dimension, float]:
    """
    Parse a unit string into (dimension, factor_to_base).

    Mass-per-area is reduced to g/m²; length to m; counts per m².
    """
    raw = symbol.strip()
    norm = _normalize_unit_text(raw)

    if norm == "m":
        return _Dimension(length_exp=Fraction(1)), 1.0

    if "dried soil" in norm or norm == "mg/kg":
        return _Dimension(mass_exp=Fraction(0), substance="soil_ratio"), 1.0

    # Normalize to explicit / form
    text = norm.replace(" ha-1", "/ha").replace(" m-2", "/m2")
    text = text.replace(" m-1", "/m")
    text = re.sub(r"\s+", " ", text)

    numer: list[str] = []
    denom: list[str] = []
    for seg_i, segment in enumerate(text.split("/")):
        chunk = [c for c in segment.strip().split() if c and c not in {"or", "per"}]
        if seg_i == 0:
            numer.extend(chunk)
        else:
            denom.extend(chunk)

    # Handle trailing ha-1 style already merged; also "kg ha-1" as two tokens in numer
    expanded_numer: list[str] = []
    expanded_denom: list[str] = []
    for side, bucket in ((numer, expanded_numer), (denom, expanded_denom)):
        for tok in side:
            if tok.endswith("-1") and len(tok) > 2:
                expanded_denom.append(tok[:-2])
            elif tok == "ha1":  # shouldn't happen
                expanded_denom.append("ha")
            else:
                bucket.append(tok)
    numer, denom = expanded_numer, expanded_denom

    # "kg ha-1" pattern: second token ha with -1 already handled via split
    # Re-split original on whitespace for "kg N ha-1"
    if not denom and " " in norm:
        bits = norm.split()
        idx = 0
        numer, denom = [], []
        while idx < len(bits):
            b = bits[idx]
            if b in {"ha-1", "m-2", "m-1"} or (b.endswith("-1") and b[:-2] in {"ha", "m2", "m"}):
                base = b[:-2] if b.endswith("-1") else b.replace("-1", "")
                denom.append(base if base != "m" else "m2" if b == "m-2" else "m")
                idx += 1
            elif idx + 1 < len(bits) and bits[idx + 1] in {"ha-1", "m-2"}:
                numer.append(b)
                denom.append(bits[idx + 1].replace("-1", ""))
                idx += 2
            else:
                numer.append(b)
                idx += 1

    substance: Optional[str] = None
    mass_n = mass_d = 0
    area_n = area_d = 0
    len_n = len_d = 0
    count_n = count_d = 0
    factor = 1.0

    def add_mass(scale: float, den: bool) -> None:
        nonlocal factor, mass_n, mass_d
        if den:
            mass_d += 1
            factor /= scale
        else:
            mass_n += 1
            factor *= scale

    def add_area(scale: float, den: bool) -> None:
        nonlocal factor, area_n, area_d
        if den:
            area_d += 1
            factor /= scale
        else:
            area_n += 1
            factor *= scale

    def process(tok: str, den: bool) -> None:
        nonlocal substance, count_n, count_d, len_n, len_d, factor
        if tok in _MASS_TO_G:
            add_mass(_MASS_TO_G[tok], den)
        elif tok in {"n", "p", "k", "dm"}:
            tag = tok.upper()
            substance = tag if substance is None else f"{substance}_{tag}"
        elif tok == "p2o5":
            substance = "P2O5" if substance is None else f"{substance}_P2O5"
        elif tok == "k2o":
            substance = "K2O" if substance is None else f"{substance}_K2O"
        elif tok in {"diammonium", "phosphate"}:
            substance = "diammonium_phosphate"
        elif tok == "ha":
            add_area(_AREA_TO_M2["ha"], den)
        elif tok in {"m2", "m-2"}:
            add_area(1.0, den)
        elif tok == "m":
            if den:
                len_d += 1
                factor /= _LENGTH_TO_M["m"]
            else:
                len_n += 1
                factor *= _LENGTH_TO_M["m"]
        elif tok in _COUNT_BASES or tok.rstrip("s") in _COUNT_BASES:
            if den:
                count_d += 1
            else:
                count_n += 1
        elif tok in {"row", "dried", "soil", "5"}:
            return
        elif tok == "kg" and den:
            add_mass(_MASS_TO_G["kg"], True)
        else:
            raise ValueError(f"Unrecognized unit token {tok!r} in {symbol!r}")

    for t in numer:
        process(t, False)
    for t in denom:
        process(t, True)

    dim = _Dimension(
        mass_exp=Fraction(mass_n - mass_d, 1),
        length_exp=Fraction(2 * (area_n - area_d) + (len_n - len_d), 1),
        count_exp=Fraction(count_n - count_d, 1),
        substance=substance,
    )
    return dim, factor


class Unit:
    """
    A single measurable unit with dimensional analysis and conversion.

    Use scalar multiplication to build a ``Quantity``::

        >>> q1 = 50 * unit("kg ha-1")
        >>> q2 = 50000 * unit("g ha-1")
        >>> q1 == q2
        True
    """

    __slots__ = ("_symbol", "_dim", "_factor")

    def __init__(self, symbol: str) -> None:
        self._symbol = symbol.strip()
        self._dim, self._factor = _parse_unit_symbol(self._symbol)

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def dimension(self) -> _Dimension:
        return self._dim

    def is_compatible(self, other: Unit) -> bool:
        return self._dim.is_compatible(other._dim)

    def conversion_factor_to(self, other: Unit) -> float:
        """
        Multiply a magnitude in ``self`` by this factor to express it in ``other``.

        Raises ``ValueError`` when dimensions differ.
        """
        if not self.is_compatible(other):
            raise ValueError(
                f"Incompatible units: {self._symbol!r} ({self._dim}) vs "
                f"{other._symbol!r} ({other._dim})"
            )
        return self._factor / other._factor

    def __mul__(self, magnitude: Number) -> Quantity:
        if isinstance(magnitude, Unit):
            return NotImplemented
        return Quantity(float(magnitude), self)

    def __rmul__(self, magnitude: Number) -> Quantity:
        return Quantity(float(magnitude), self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Unit):
            return NotImplemented
        return self.is_compatible(other) and abs(self.conversion_factor_to(other) - 1.0) < 1e-12

    def __repr__(self) -> str:
        return f"Unit({self._symbol!r})"


@dataclass
class Quantity:
    """A numeric magnitude with a unit (e.g. 50 kg ha⁻¹)."""

    magnitude: float
    unit: Unit

    def to(self, target: Union[Unit, str]) -> Quantity:
        if isinstance(target, str):
            target = Unit(target)
        if self.unit == target:
            return Quantity(self.magnitude, self.unit)
        factor = self.unit.conversion_factor_to(target)
        return Quantity(self.magnitude * factor, target)

    def __mul__(self, other: Number) -> Quantity:
        if isinstance(other, (int, float)):
            return Quantity(self.magnitude * float(other), self.unit)
        return NotImplemented

    def __rmul__(self, other: Number) -> Quantity:
        if isinstance(other, (int, float)):
            return Quantity(self.magnitude * float(other), self.unit)
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quantity):
            return NotImplemented
        if not self.unit.is_compatible(other.unit):
            return False
        return abs(self.magnitude - other.to(self.unit).magnitude) < 1e-9 * max(
            1.0, abs(self.magnitude), abs(other.magnitude)
        )

    def __repr__(self) -> str:
        return f"Quantity({self.magnitude}, {self.unit.symbol!r})"


def unit(symbol: str) -> Unit:
    """Shortcut for ``Unit(symbol)``."""
    return Unit(symbol)


__all__ = [
    "Unit",
    "UnitStorage",
    "Quantity",
    "unit",
    "ValueUnitGroup",
    "WOPKE_100_VALUE_UNIT_GROUPS",
    "WOPKE_100_GT_VALUE_UNIT_GROUPS",
    "VALUE_TO_UNIT_FIELD",
    "UNIT_FIELDS",
    "value_unit_groups",
    "unit_field_for_value",
]
