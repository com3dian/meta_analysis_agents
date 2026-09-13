Context

dataset -> 3 approaches
evaluation pipeline


# Evaluation

The outputs of all approaches are saved in CSV format, the same as the ground-truth labels. We classify the mismatches between the predictions and ground truth into three types, each measured by a corresponding metric:

1. **Incorrect:** The prediction record matches a ground-truth record, but the values in one or more fields are incorrect. This type of error is measured by the **similarity score** (0 to 1).
2. **Hallucination:** The prediction contains records that are not present in the ground truth. This type of error is measured by the **factual rate** (0 to 1).
3. **Incompleteness:** The prediction does not contain records that exist in the ground truth. This type of error is measured by the **completeness rate** (0 to 1).

We then design a similarity measure function that takes pairs of records from the prediction and ground truth and produces a match score from 0 to 1, representing how similar the two records are. A score of 1 means the records are identical, while 0 means they are totally different.

Specifically, the similarity is computed as follows: for normal fields, we perform pairwise comparison; for text fields, we use ROUGE-L F1 on characters; for numbers, we use numerical matching with a **2% relative threshold** vs ground truth (`|pred − gt| / |gt| ≤ 0.02`); if a numerical value contains a unit, we convert to a common scale then apply the same 2% threshold; for dates, we use date matching with rule-based matching; for boolean fodder flags, we normalize Yes/No / 1/0 / true/false then exact-match.

Given the output of an LLM, we perform the evaluation under the following pipeline. First, we identify papers that have no output for each method. This could be because the final output is not parsable as JSON structured output, or because the method takes much longer to produce an output. Suppose we have \(N_{valid}\) valid outputs out of \(N_{total}\) paper inputs. For each valid output, we take the records in the CSV and perform greedy matching between the output records and the ground-truth records based on the record similarity score (explained later).

After performing the greedy matching, ideally, the records in the method output and the ground truth should have a one-to-one match. If not, we handle the following cases:

1. **Two or more predictions for one ground-truth record:**  
   Suppose the two predictions are \(\{pred_A, pred_B, \ldots\}\), and their match scores with a ground-truth record \(gt_1\) are \(S(pred_A, gt_1)\) and \(S(pred_B, gt_1)\), where \(S(pred_A, gt_1) > S(pred_B, gt_1)\). We match \(pred_A\) with the ground-truth record \(gt_1\). If \(pred_B\) has another match score with \(gt_2\) that is its highest score, then we match \(pred_B\) with \(gt_2\).

2. **Leftover records:**  
   After performing this matching, if there are leftover records, we classify them as either **hallucination** (leftover records from the prediction) or **incompleteness** (leftover records from the ground truth).


## Unit columns and canonical mapping

Prediction CSVs (42-field schema) carry **two shared unit columns per record**:

- `N Unit` — applies to all `N input *` / `N total in IC` fields
- `Yield unit` — applies to all `unified yield *` fields

Ground truth uses the same yield unit column (`Yield unit`) plus separate fertilizer unit columns (`Unit.1` = N, `Unit.2` = P, `Unit.3` = K) and `Unit of density`.

During scoring, `src/experimentutils/units.py` canonicalizes unit strings before building `Quantity` objects. Compatible units score as a unit match; values are compared after conversion to a common scale.

### What “unparsable” means

In the unit tables below, **Parser** has three values:

| Status | Meaning |
|---|---|
| `in registry` | Exact string is in `UnitStorage` (observed in GT or predictions). |
| `parsable` | Not in the registry, but `Unit(...)` understands it after `sanitize_unit_string()`. |
| `unparsable` | `Unit(...)` still fails → eval falls back to **character-level ROUGE-L** on the raw unit strings (no dimensional conversion, usually scores 0 unless strings are nearly identical). |

**Parsable** units participate in quantity-aware scoring (e.g. `5 g N m-2` vs `50 kg/ha` → match). **Unparsable** units do not.

### Unit sanitization (`sanitize_unit_string`)

Before parsing, eval normalizes common LLM output variants in `src/experimentutils/units.py`:

| Pattern in model output | Normalized to | Notes |
|---|---|---|
| Unicode hyphens / thin spaces / superscripts (`ha‑1`, `m⁻²`, `g m⁻²`) | ASCII `ha-1`, `m-2` | PDF / copy-paste artifacts |
| LaTeX exponents (`ha^{-1}`, `m^-2`) | `ha-1`, `m-2` | |
| Glued tokens (`kgNha-1`, `kgha-1`, `tha-1`, `gNm-2`) | `kg N ha-1`, `kg ha-1`, `t ha-1`, `g N m-2` | |
| Reordered N tag (`kg/ha N`, `kg / haN`) | `kg N/ha` | |
| Tonne synonyms (`tonnes/ha`, `ton/ha`, `MT/ha`, bare `ton`) | `t/ha` or `t ha-1` | |
| `dry matter`, `dry weight` | `DM` | then yield rules strip DM tag |
| `g per m²`, middle dots (`kg·ha−1`) | `g/m2`, `kg ha-1` | |
| Typos (`kg Nh a-1`, `kg NP ha-1`) | `kg N ha-1` | NP treated as N for scoring |
| Parenthetical annotations (`kg ha⁻¹ (TDMY)`) | stripped | |

**Still unparsable** (need value rescaling or are not mass/area units — cannot fix with string rules alone):

| Unit string (examples) | Why |
|---|---|
| `g/20 m2`, `g/20 m²` | yield is per 20 m² plot, not per m² |
| `kg/6.1 m of row`, `kg per 6.1 m row` | per-row basis, not per ha |
| `kg per m² (potato) and g per m² (bean)` | multi-crop composite label |
| `percentage of sole crop` | relative %, not a physical unit |
| `280g of Nkg-1` | garbled free text |

After sanitization (current outputs): **N Unit** — 9,380 / 9,382 rows parsable; **Yield unit** — 10,220 / 10,445 rows parsable.

Tables below list units **as stored** in CSVs; many rows marked `unparsable` in the tables are now handled by `sanitize_unit_string()` at eval time.

Tables below were generated by scanning **all** `outputs/*/by_paper/*.csv` prediction files (`1345` files) and the ground-truth spreadsheet `data/wopke_paper_code/Database for combined sample 2015-03-05.csv`.

### Canonicalization rules (summary)

| Group | Rule | Example |
|---|---|---|
| N fertilizer (GT) | bare `kg/ha` / `kg ha-1` variants without an `N` tag → `kg N/ha` | `kg/ha` + `50` ↔ `5 g N m-2` |
| N fertilizer (pred) | same bare-`kg/ha` expansion when nutrient context is N | `kg/ha` → `kg N/ha` |
| Yield (pred) | `g DM m-2`, `kg DM ha-1`, `Mg DM ha-1`, … → strip `DM` tag → `g/m2` | `473 g DM m-2` ↔ `473 g/m2` |
| Yield (GT) | ``unified yield *`` values are **t/ha scale**; mislabeled ``g/m2`` / ``kg/ha`` / ``q/ha`` / ``kg/m2`` → ``t/ha`` for comparison (``t/ha`` and ``Mg/ha`` labels kept as-is) | GT `3.32` + `g/m2` ↔ pred `332` + `g DM m-2`; GT `3.93` + `kg/ha` ↔ pred `3.93` + `t/ha` |
| Cross-scale equivalences | parser converts compatible mass/area units | `1 t/ha = 100 g/m²`; `1 kg N/ha = 10 g N/m²` |

**Note:** GT `g/m2` / `kg/ha` / etc. yield rows also have legacy `Yield sc *` columns with values on the label’s true scale. Eval scores `unified yield *` only, using the t/ha reinterpretation above.

### Ground truth

#### Density (GT only)

| Unit (as stored) | Count | Canonical for eval | Parser | Mapping |
|---|---:|---|---|---|
| `plants/m2` | 362 | `plants/m2` | in registry | GT only; not unit-scored in 42-field prediction eval |
| `plants/ha` | 206 | `plants/ha` | in registry | GT only; not unit-scored in 42-field prediction eval |
| `seeds/m2` | 47 | `seeds/m2` | in registry | GT only; not unit-scored in 42-field prediction eval |
| `kg/ha` | 37 | `kg/ha` | in registry | GT only; not unit-scored in 42-field prediction eval |
| `grains/m2` | 7 | `grains/m2` | in registry | GT only; not unit-scored in 42-field prediction eval |
| `kg seeds/ha` | 7 | `kg seeds/ha` | in registry | GT only; not unit-scored in 42-field prediction eval |
| `plants or g/m2` | 2 | `plants or g/m2` | in registry | GT only; not unit-scored in 42-field prediction eval |
| `seedlings/m2` | 2 | `seedlings/m2` | in registry | GT only; not unit-scored in 42-field prediction eval |
| `seeds or plants/m2` | 2 | `seeds or plants/m2` | in registry | GT only; not unit-scored in 42-field prediction eval |
| `seeds/5 m row` | 1 | `seeds/5 m row` | in registry | GT only; not unit-scored in 42-field prediction eval |

#### N fertilizer

| Unit (as stored) | Count | Canonical for eval | Parser | Mapping |
|---|---:|---|---|---|
| `kg N/ha` | 525 | `kg N/ha` | in registry | unchanged |
| `kg/ha` | 35 | `kg N/ha` | in registry | bare mass/ha without nutrient tag → `kg N/ha` |
| `Kg N/ha` | 33 | `Kg N/ha` | in registry | unchanged |

#### P fertilizer (GT only)

| Unit (as stored) | Count | Canonical for eval | Parser | Mapping |
|---|---:|---|---|---|
| `kg P/ha` | 158 | `kg P/ha` | in registry | unchanged |
| `kg P2O5/ha` | 46 | `kg P2O5/ha` | in registry | unchanged |
| `kg/ha` | 21 | `kg P/ha` | in registry | bare mass/ha without nutrient tag → `kg P/ha` |
| `mg/kg dried soil` | 2 | `mg/kg dried soil` | in registry | unchanged |
| `kg P2O5 /ha` | 1 | `kg P2O5 /ha` | in registry | unchanged |
| `kg diammonium phosphate/ha` | 1 | `kg diammonium phosphate/ha` | in registry | unchanged |

#### K fertilizer (GT only)

| Unit (as stored) | Count | Canonical for eval | Parser | Mapping |
|---|---:|---|---|---|
| `kg K/ha` | 111 | `kg K/ha` | in registry | unchanged |
| `kg K2O/ha` | 21 | `kg K2O/ha` | in registry | unchanged |
| `Kg K/ha` | 12 | `Kg K/ha` | in registry | unchanged |
| `kg/ha` | 2 | `kg K/ha` | in registry | bare mass/ha without nutrient tag → `kg K/ha` |
| `mg/kg dried soil` | 2 | `mg/kg dried soil` | in registry | unchanged |

#### Yield

| Unit (as stored) | Count | Canonical for eval | Parser | Mapping |
|---|---:|---|---|---|
| `kg/ha` | 376 | `kg/ha` | in registry | unchanged |
| `t/ha` | 203 | `t/ha` | in registry | unchanged |
| `g/m2` | 87 | `t/ha` | in registry | GT label is `g/m2`, but `unified yield *` values are stored in t/ha (×100 vs g/m²) |
| `q/ha` | 34 | `q/ha` | in registry | unchanged |
| `Mg/ha` | 30 | `Mg/ha` | in registry | unchanged |
| `kg/m2` | 16 | `kg/m2` | in registry | unchanged |

### Model outputs

#### N fertilizer

| Unit (as stored) | Count | Canonical for eval | Parser | Mapping |
|---|---:|---|---|---|
| `kg N ha-1` | 4306 | `kg N ha-1` | in registry | unchanged |
| `kg N ha−1` | 1553 | `kg N ha−1` | in registry | unchanged |
| `kg ha-1` | 837 | `kg ha-1` | in registry | unchanged |
| `kg N ha⁻¹` | 386 | `kg N ha⁻¹` | in registry | unchanged |
| `kg/ha` | 384 | `kg N/ha` | in registry | bare mass/ha without nutrient tag → `kg N/ha` |
| `g N m-2` | 330 | `g N m-2` | in registry | unchanged |
| `kg N/ha` | 197 | `kg N/ha` | in registry | unchanged |
| `kg ha⁻¹` | 160 | `kg N/ha` | in registry | bare mass/ha without nutrient tag → `kg N/ha` |
| `kg ha−1` | 155 | `kg ha−1` | in registry | unchanged |
| `kg N ha‑1` | 108 | `kg N ha‑1` | unparsable | unchanged |
| `kg ha‑1` | 73 | `kg ha‑1` | unparsable | unchanged |
| `kgN / ha` | 68 | `kgN / ha` | unparsable | unchanged |
| `kg N ha^{-1}` | 67 | `kg N ha^{-1}` | unparsable | unchanged |
| `kgNha-1` | 60 | `kgNha-1` | unparsable | unchanged |
| `g N m⁻²` | 51 | `g N m⁻²` | unparsable | unchanged |
| `kg` | 51 | `kg` | parsable | unchanged |
| `kg N` | 51 | `kg N` | parsable | unchanged |
| `g N m^-2` | 48 | `g N m^-2` | unparsable | unchanged |
| `mg kg-1` | 47 | `mg kg-1` | parsable | unchanged |
| `kg / haN` | 42 | `kg / haN` | unparsable | unchanged |
| `kg/ha N` | 40 | `kg/ha N` | unparsable | unchanged |
| `g N m‑2` | 30 | `g N m‑2` | unparsable | unchanged |
| `kg NP ha-1` | 29 | `kg NP ha-1` | unparsable | unchanged |
| `kg ha-1 N` | 27 | `kg ha-1 N` | parsable | unchanged |
| `kgN` | 24 | `kgN` | unparsable | unchanged |
| `kg / ha` | 23 | `kg N/ha` | parsable | bare mass/ha without nutrient tag → `kg N/ha` |
| `kg N P ha-1` | 21 | `kg N P ha-1` | parsable | unchanged |
| `kgNha^{-1}` | 21 | `kgNha^{-1}` | unparsable | unchanged |
| `t/ha` | 20 | `t/ha` | in registry | unchanged |
| `kg ha‑1` | 19 | `kg ha‑1` | unparsable | unchanged |
| `kg N ha-1 year-1` | 18 | `kg N ha-1 year-1` | unparsable | unchanged |
| `N kg ha-1` | 16 | `N kg ha-1` | parsable | unchanged |
| `kg ha⁻¹ N` | 12 | `kg ha⁻¹ N` | parsable | unchanged |
| `kg P ha-1` | 10 | `kg P ha-1` | in registry | unchanged |
| `kg N P ha‑1` | 9 | `kg N P ha‑1` | unparsable | unchanged |
| `gNm-2` | 8 | `gNm-2` | unparsable | unchanged |
| `kg N / ha` | 8 | `kg N / ha` | parsable | unchanged |
| `kg P2O5 ha⁻¹` | 8 | `kg P2O5 ha⁻¹` | parsable | unchanged |
| `kg·ha−1` | 8 | `kg·ha−1` | unparsable | unchanged |
| `N` | 7 | `N` | parsable | unchanged |
| `kg N ha−1 year−1` | 6 | `kg N ha−1 year−1` | unparsable | unchanged |
| `kg Nh a-1` | 6 | `kg Nh a-1` | unparsable | unchanged |
| `kg P2O5 ha-1` | 5 | `kg P2O5 ha-1` | in registry | unchanged |
| `kg·ha⁻¹` | 5 | `kg·ha⁻¹` | unparsable | unchanged |
| `g/plant` | 4 | `g/plant` | parsable | unchanged |
| `kg ha-1 nitrogen` | 4 | `kg ha-1 nitrogen` | unparsable | unchanged |
| `kg ha−1 N` | 4 | `kg ha−1 N` | parsable | unchanged |
| `kg · ha⁻1` | 4 | `kg · ha⁻1` | unparsable | unchanged |
| `mg kg−1` | 4 | `mg kg−1` | parsable | unchanged |
| `g N m−2` | 3 | `g N m−2` | parsable | unchanged |
| `kg ha-1 (8-8-8% NPK)` | 3 | `kg ha-1 (8-8-8% NPK)` | unparsable | unchanged |
| `280g of Nkg-1` | 2 | `280g of Nkg-1` | unparsable | unchanged |

#### Yield

| Unit (as stored) | Count | Canonical for eval | Parser | Mapping |
|---|---:|---|---|---|
| `kg ha-1` | 2274 | `kg ha-1` | in registry | unchanged |
| `t/ha` | 1614 | `t/ha` | in registry | unchanged |
| `kg/ha` | 1031 | `kg/ha` | in registry | unchanged |
| `t ha-1` | 705 | `t ha-1` | in registry | unchanged |
| `g m-2` | 693 | `g m-2` | in registry | unchanged |
| `Mg ha-1` | 541 | `Mg ha-1` | in registry | unchanged |
| `q/ha` | 498 | `q/ha` | in registry | unchanged |
| `g DM m-2` | 401 | `g/m2` | in registry | strip DM tag; compare as g/m² |
| `g/m2` | 262 | `g/m2` | in registry | unchanged |
| `tonnes/ha` | 261 | `tonnes/ha` | unparsable | unchanged |
| `kg ha⁻¹` | 222 | `kg ha⁻¹` | in registry | unchanged |
| `kg ha−1` | 222 | `kg ha−1` | in registry | unchanged |
| `t ha−1` | 123 | `t ha−1` | in registry | unchanged |
| `g plant-1` | 112 | `g plant-1` | parsable | unchanged |
| `MT/ha` | 108 | `MT/ha` | unparsable | unchanged |
| `t ha⁻¹` | 94 | `t ha⁻¹` | in registry | unchanged |
| `ton/ha` | 76 | `ton/ha` | unparsable | unchanged |
| `kg ha^{-1}` | 69 | `kg ha^{-1}` | unparsable | unchanged |
| `t ha‑1` | 65 | `t ha‑1` | unparsable | unchanged |
| `g dry matter/m2` | 60 | `g dry matter/m2` | unparsable | unchanged |
| `Mg DM ha-1` | 56 | `g/m2` | parsable | strip DM tag; compare as g/m² |
| `kg ha‑1` | 54 | `kg ha‑1` | unparsable | unchanged |
| `kg/6.1 m of row` | 53 | `kg/6.1 m of row` | unparsable | unchanged |
| `g/20 m²` | 50 | `g/20 m²` | unparsable | unchanged |
| `t ha⁻¹` | 45 | `t ha⁻¹` | parsable | unchanged |
| `Mg ha⁻¹` | 41 | `Mg ha⁻¹` | in registry | unchanged |
| `g DM m‑2` | 35 | `g DM m‑2` | unparsable | unchanged |
| `g m⁻²` | 35 | `g m⁻²` | unparsable | unchanged |
| `g DM m⁻²` | 34 | `g DM m⁻²` | unparsable | unchanged |
| `g m⁻²` | 34 | `g m⁻²` | unparsable | unchanged |
| `g/20 m2` | 32 | `g/20 m2` | unparsable | unchanged |
| `kg DM ha-1` | 32 | `g/m2` | parsable | strip DM tag; compare as g/m² |
| `kg N ha-1` | 32 | `kg N ha-1` | in registry | unchanged |
| `Mg ha−1` | 31 | `Mg ha−1` | in registry | unchanged |
| `kg per 6.1 m row` | 28 | `kg per 6.1 m row` | unparsable | unchanged |
| `g m^-2` | 24 | `g m^-2` | unparsable | unchanged |
| `kg ha⁻¹` | 23 | `kg ha⁻¹` | parsable | unchanged |
| `kg m-2` | 22 | `kg m-2` | parsable | unchanged |
| `Mg ha‑1` | 21 | `Mg ha‑1` | unparsable | unchanged |
| `tha-1` | 21 | `tha-1` | unparsable | unchanged |
| `ton` | 20 | `ton` | unparsable | unchanged |
| `t ha‑1` | 20 | `t ha‑1` | unparsable | unchanged |
| `kg ha‑1` | 19 | `kg ha‑1` | unparsable | unchanged |
| `g/20m2` | 17 | `g/20m2` | unparsable | unchanged |
| `ton ha-1` | 16 | `ton ha-1` | unparsable | unchanged |
| `kg seeds per ha` | 13 | `kg seeds per ha` | parsable | unchanged |
| `g N m-2` | 12 | `g N m-2` | in registry | unchanged |
| `kg per m² (potato) and g per m² (bean)` | 12 | `kg per m² (potato) and g per m² (bean)` | unparsable | unchanged |
| `Mg DM ha⁻¹` | 10 | `g/m2` | parsable | strip DM tag; compare as g/m² |
| `g / 20m2` | 10 | `g / 20m2` | unparsable | unchanged |
| `g m-1` | 9 | `g m-1` | parsable | unchanged |
| `g/m²` | 8 | `g/m²` | parsable | unchanged |
| `g DM m⁻²` | 8 | `g DM m⁻²` | unparsable | unchanged |
| `g N m⁻²` | 8 | `g N m⁻²` | unparsable | unchanged |
| `kg (potato) / g (bean)` | 8 | `kg (potato) / g (bean)` | unparsable | unchanged |
| `kg ha⁻¹ (TDMY)` | 8 | `kg ha⁻¹ (TDMY)` | unparsable | unchanged |
| `kg/m² (Potato), g/m² (Bean)` | 8 | `kg/m² (Potato), g/m² (Bean)` | unparsable | unchanged |
| `kgha-1` | 8 | `kgha-1` | unparsable | unchanged |
| `g plant⁻¹` | 7 | `g plant⁻¹` | unparsable | unchanged |
| `g plant⁻¹` | 7 | `g plant⁻¹` | unparsable | unchanged |
| `t DM ha-1` | 7 | `g/m2` | parsable | strip DM tag; compare as g/m² |
| `g m-1 per 2 rows` | 6 | `g m-1 per 2 rows` | unparsable | unchanged |
| `g plant‑1` | 6 | `g plant‑1` | unparsable | unchanged |
| `percentage of sole crop` | 6 | `percentage of sole crop` | unparsable | unchanged |
| `g m‑2` | 5 | `g m‑2` | unparsable | unchanged |
| `dry matter` | 4 | `dry matter` | unparsable | unchanged |
| `dry weight` | 4 | `dry weight` | unparsable | unchanged |
| `g m‑2` | 4 | `g m‑2` | unparsable | unchanged |
| `g plant−1` | 4 | `g plant−1` | parsable | unchanged |
| `kg DM ha‑1` | 4 | `kg DM ha‑1` | unparsable | unchanged |
| `Mgha-1` | 3 | `Mgha-1` | unparsable | unchanged |
| `g m‑1` | 3 | `g m‑1` | unparsable | unchanged |
| `g m−1 per 2 rows` | 3 | `g m−1 per 2 rows` | unparsable | unchanged |
| `g m−2` | 3 | `g m−2` | parsable | unchanged |
| `kg / ha` | 3 | `kg / ha` | parsable | unchanged |
| `Mg ha^{-1}` | 2 | `Mg ha^{-1}` | unparsable | unchanged |
| `Mg/ha` | 2 | `Mg/ha` | in registry | unchanged |
| `g` | 2 | `g` | parsable | unchanged |
| `g per m²` | 2 | `g per m²` | unparsable | unchanged |
| `kg` | 2 | `kg` | parsable | unchanged |
| `kg per m²` | 2 | `kg per m²` | unparsable | unchanged |
| `kg/m²` | 2 | `kg/m²` | parsable | unchanged |
| `kg DM ha⁻¹` | 1 | `g/m2` | parsable | strip DM tag; compare as g/m² |
| `kg N ha⁻¹` | 1 | `kg N ha⁻¹` | in registry | unchanged |
| `t DM ha−1` | 1 | `g/m2` | parsable | strip DM tag; compare as g/m² |
| `t N ha−1` | 1 | `t N ha−1` | parsable | unchanged |

