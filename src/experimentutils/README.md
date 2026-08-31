# experimentutils

Utilities for running and evaluating meta-analysis extraction experiments (wopke_100 corpus).

## Module map


| Module              | Purpose                                                                                                               |
| ------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `eval_utils.py`     | **Dataset validation & scoring** — GT loading, field similarity, record matching, swap logic, metrics, debug printers |
| `units.py`          | Unit registry (`UnitStorage`), value+unit field pairs (`ValueUnitGroup`), dimensional arithmetic (`Unit`, `Quantity`) |
| `file_utils.py`     | Paper paths, markdown/PDF I/O                                                                                         |
| `output_utils.py`   | Dated CSV/JSON output paths                                                                                           |
| `standard_utils.py` | Filter `METADATA_STANDARDS` schemas                                                                                   |
| `progress_utils.py` | Orchestrator progress bars                                                                                            |


## Where is “dataset validation” defined?

There is no single function named `dataset_validation`. Scoring and validation against ground truth live in `**eval_utils.py`**:


| Function                                                 | Role                                                                          |
| -------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `field_similarity_score`                                 | Per-cell similarity in `[0, 1]` (numeric, year, text/ROUGE-L)                 |
| `evaluate_record_pair_with_units`                        | Two records: per-field scores, value+unit, crop swap → `RecordPairEvaluation` |
| `score_field_between_records`                            | Single field between two records (routes to value+unit or text)               |
| `find_crop_swap_pairs`                                   | Detect column pairs ending in `1` / `2` (crop swap)                           |
| `swap_1_2_fields_for_records`                            | Apply swap to all prediction rows                                             |
| `choose_best_global_swap_orientation`                    | Pick original vs globally swapped predictions (paper-level)                   |
| `_pair_scores` / `evaluate_record_pair`                  | Score one pred row vs one GT row; **per-field swap** inside the pair          |
| `match_records_greedy`                                   | Greedy 1:1 row matching (value similarity + swap)                             |
| `match_records_greedy_presence`                          | Same, but presence/label signal (used in notebooks)                           |
| `overall_metrics_presence` / `confusion_counts_presence` | TP/FP/FN/TN + `tp_avg_similarity`                                             |
| `evaluate_method_scores`                                 | Normalised score over matched pairs                                           |
| `print_matching_pairs` / `print_matching_pairs_from_df`  | CLI table: FIELD / EXTRACTED / GT / SCORE                                     |
| `print_field_value_pairs_vs_gt`                          | Step-1 artifact vs GT (no row alignment)                                      |


**Notebooks** often duplicate a subset of this logic locally. For wopke per-paper tables see:

- `notebooks/test_stuff/per_paper_results_table.ipynb` — `show_paper`, `show_paper_record_pairs`, local `confusion_counts` / `overall_metrics`
- `notebooks/test_stuff/test_validation.ipynb` — quick tests of `field_similarity_score` / `units`

**Unrelated:** `src/static_workflow/two_step_text_to_dataset.py` defines `validate_schema_records_dataset` / `validate_fact_dataset` for workflow **output shape**, not GT comparison.

---

## Value + unit field pairs (wopke_100)

Numeric amounts and yields are stored as **separate value and unit columns** on each record. Unit-aware scoring should combine them before comparison (see `units.py`).

Defined in code as `WOPKE_100_VALUE_UNIT_GROUPS` in `units.py`:

### Prediction / 42-field schema


| Unit column  | Value columns (same row)                                                               |
| ------------ | -------------------------------------------------------------------------------------- |
| `N Unit`     | `N input SC1`, `N input SC2`, `N input IC1`, `N input IC2`, `N total in IC`            |
| `P Unit`     | `P input SC1`, `P input SC2`, `P input IC1`, `P input IC2`, `P total in IC`            |
| `K Unit`     | `K input SC1`, `K input SC2`, `K input IC1`, `K input IC2`, `K total in IC`            |
| `Yield unit` | `unified yield sc 1`, `unified yield sc 2`, `unified yield ic 1`, `unified yield ic 2` |


Example (one record):

```
N input SC1 = 120,  N input IC1 = 60,  N Unit = kg N ha-1
→ compare as Quantity(120, kg N ha-1) vs Quantity(60, kg N ha-1)  (future)

unified yield sc 1 = 8.5,  Yield unit = t/ha
→ compare as Quantity(8.5, t/ha)
```

### Ground truth (`Database for combined sample 2015-03-05.csv`)

Default path: `data/wopke_paper_code/Database for combined sample 2015-03-05.csv` (same 100-study / 746-row table as `wopke100.xlsx`, except the data-location header).

Duplicate headers are deduplicated when loaded (`load_ground_truth`). Standard names that differ from the file are **copied** via `GT_COLUMN_ALIASES` (original columns kept):

| GT column       | Standard field      |
| --------------- | ------------------- |
| `N sc 1`        | `Replications SC1`  |
| `N sc 2`        | `Replications SC2`  |
| `N ic 1`        | `Replications IC1`  |
| `N ic 2`        | `Replications IC2`  |
| `Unit.1`        | `N Unit`            |
| `Unit.2`        | `P Unit`            |
| `Unit.3`        | `K Unit`            |
| `Data location` | `Data source`       |

Unit columns map as:

| GT unit column    | Value columns                                                            |
| ----------------- | ------------------------------------------------------------------------ |
| `Unit.1`          | Same N input fields as above                                             |
| `Unit.2`          | Same P input fields                                                      |
| `Unit.3`          | Same K input fields                                                      |
| `Unit of density` | `Density ic 1`, `Density ic 2`, `Density sc 1`, `Density sc 2` (GT only) |
| `Yield unit`      | Same unified yield fields                                                |

Use `value_unit_groups(gt_spreadsheet=True)` for GT groups. `wopke_100_shared_fields(gt.columns)` is the standard ∩ GT intersection used for scoring.

### Helpers

```python
from src.experimentutils.units import (
    VALUE_TO_UNIT_FIELD,
    unit_field_for_value,
    WOPKE_100_VALUE_UNIT_GROUPS,
)

unit_field_for_value("N input SC1")   # → "N Unit"
unit_field_for_value("unified yield ic 2")  # → "Yield unit"
```

(`eval_utils` re-exports these for backward compatibility.)

---

## Crop 1/2 swap (current behaviour)

When the model swaps crop species 1 and 2, **all** detected `…1` / `…2` column pairs are swapped together (`find_crop_swap_pairs`):

- Species, types, fodder flags, densities, N/P/K inputs, unified yields, sowing/harvest dates, replications, PLER, etc.
- **Not swapped:** `N Unit`, `P Unit`, `K Unit`, `Yield unit` (one unit per record for the whole group)

Swap is applied at two levels today:

1. **Per record pair** — `_pair_scores` tries original vs swapped extraction row when matching to a GT row.
2. **Per paper (notebook)** — `choose_best_global_swap_orientation` swaps all prediction rows and picks the orientation with higher `tp_avg_similarity`.

---

## Two-record evaluation (`evaluate_record_pair_with_units`)

Compare one prediction row vs one GT row (dict or pandas Series):

1. **Per field** — `score_all_fields_between_records` fills a score dict.
2. **Value + unit** — fields in `VALUE_TO_UNIT_FIELD` use `Quantity(value, unit)` equality. When that score is `1.0`, the paired unit column in `fields` also gets `1.0` (not a separate text match on `kg ha-1` vs `g ha-1`).
3. **Crop swap** — swap prediction `…1` / `…2` columns, re-score, keep the orientation with higher **mean** score.

```python
from src.experimentutils.eval_utils import evaluate_record_pair_with_units

result = evaluate_record_pair_with_units(pred_row, gt_row, shared_fields)
result.field_scores          # final (best orientation)
result.field_scores_original
result.field_scores_swapped
result.swapped
result.mean_score            # mean of field_scores
result.mean_score_original
result.mean_score_swapped
```

`_pair_scores` / `evaluate_record_pair(..., detailed=False)` use the same logic.

**Notebook follow-up:** wire `per_paper_results_table.ipynb` printed `sim` columns to `result.field_scores`.

---

## Quick imports

```python
from src.experimentutils.eval_utils import (
    field_similarity_score,
    evaluate_record_pair,
    find_crop_swap_pairs,
)
from src.experimentutils.units import (
    unit,
    Quantity,
    UnitStorage,
    WOPKE_100_VALUE_UNIT_GROUPS,
    unit_field_for_value,
)
```

Import `eval_utils` directly if the package `__init__` pulls heavy orchestrator dependencies in your environment:

```python
import importlib
eval_utils = importlib.import_module("src.experimentutils.eval_utils")
```

