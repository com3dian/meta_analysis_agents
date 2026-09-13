"""
Reproduce Yu et al. LER ~ TND mixed-model analysis from GenAI extractions.

Pipeline (per model × method):
1. Load ``outputs/<tag>/by_paper/*.csv`` (never overwrite).
2. Prepare LER / yields / TND via :mod:`ler_prep`.
3. Keep only regression-eligible rows (final LER + final TND + Study).
4. Within each Study, greedy 1:1 match eligible GenAI rows to GT rows on
   experimental-design fields only (no LER / TND / yields / dates).
   Multiple matched rows may share an ``Experimental ID``; each GT row may
   match at most one GenAI row.
5. Fit ``lme(LER ~ TND, random = ~1 | Study/Expt)`` (R ``nlme`` or Python).
6. Write checkpoint CSVs and Figure 8a-style plots (broken y-axis for extreme LER).
"""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .eval_utils import (
    load_ground_truth,
    load_ground_truth_by_study_id,
    match_records_greedy_with_units,
)
from .ler_prep import (
    exclusion_summary,
    prepare_ler_records,
)
from .papermap import FOLDER_TO_STUDY_ID

# Design-only fields for GT matching (no LER / PLER / yields / TND / dates).
MATCH_FIELDS: Tuple[str, ...] = (
    "Crop species 1",
    "Crop species 2",
    "Crop type 1",
    "Crop type 2",
    "Fodder crop 1",
    "Fodder crop 2",
    "Year of data",
    "Experimental design",
    "Intercropping pattern",
    "Density ic 1",
    "Density ic 2",
    "Density sc 1",
    "Density sc 2",
    "RDT",
    "N input SC1",
    "N input SC2",
    "N input IC1",
    "N input IC2",
    "N total in IC",
    "N Unit",
    "Data source",
)

# Main panel of Figure 8a. A broken / jumped y-axis is used only when some
# LER values are at least ``PLOT_LER_BREAK_THRESHOLD`` (default 10).
PLOT_LER_MAIN_MAX = 2.1
PLOT_LER_MAIN_MIN = 0.4
PLOT_LER_BREAK_THRESHOLD = 10.0

METHOD_PREFIXES: Dict[str, str] = {
    "direct_llm": "direct_llm",
    "workflow": "static_workflow",
    "MAS": "mas",
}
PREFIX_TO_LABEL: Dict[str, str] = {v: k for k, v in METHOD_PREFIXES.items()}

_BY_PAPER_RE = __import__("re").compile(
    r"^(?P<sid>\d+)_.+__(?P<method>direct_llm|static_workflow|mas)\.csv$"
)

PUBLISHED_TND_COEF = 0.211
MAPPED_STUDY_MAX = 90


def find_repo_root(start: Optional[Path] = None) -> Path:
    p = start or Path.cwd()
    for candidate in [p, *p.parents]:
        if (candidate / "src").is_dir() and (candidate / "outputs").is_dir():
            return candidate
    return start or Path.cwd()


def default_gt_path(repo_root: Optional[Path] = None) -> Path:
    root = repo_root or find_repo_root()
    return (
        root
        / "data"
        / "wopke_paper_code"
        / "Database for combined sample 2015-03-05.csv"
    )


def default_processed_root(repo_root: Optional[Path] = None) -> Path:
    root = repo_root or find_repo_root()
    return root / "outputs" / "ler_tnd_reproduction"


def scan_by_paper(run_dir: Path) -> Dict[str, Dict[int, Path]]:
    """Return ``{method_label: {study_id: csv_path}}``."""
    by_paper = Path(run_dir) / "by_paper"
    pred: Dict[str, Dict[int, Path]] = {label: {} for label in METHOD_PREFIXES}
    if not by_paper.is_dir():
        raise FileNotFoundError(f"No by_paper folder in {run_dir}")
    for csv_path in by_paper.glob("*.csv"):
        m = _BY_PAPER_RE.match(csv_path.name)
        if not m:
            continue
        label = PREFIX_TO_LABEL.get(m.group("method"))
        if label is None:
            continue
        pred[label][int(m.group("sid"))] = csv_path
    return pred


def list_model_tags(outputs_root: Optional[Path] = None) -> List[str]:
    root = Path(outputs_root) if outputs_root else find_repo_root() / "outputs"
    tags: List[str] = []
    if not root.is_dir():
        return tags
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        if __import__("re").match(r"^\d{4}-\d{2}-\d{2}$", run_dir.name):
            continue
        if run_dir.name == "ler_tnd_reproduction":
            continue
        if (run_dir / "by_paper").is_dir():
            tags.append(run_dir.name)
    return tags


def _read_prediction_csv(path: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df.to_dict(orient="records")


def load_and_prepare_method(
    run_dir: Path,
    method: str,
    *,
    model_tag: Optional[str] = None,
) -> pd.DataFrame:
    """Load all by-paper CSVs for one method and prepare LER/TND fields."""
    if method not in METHOD_PREFIXES:
        raise ValueError(f"Unknown method {method!r}; choose from {list(METHOD_PREFIXES)}")
    pred_map = scan_by_paper(run_dir).get(method, {})
    records: List[Dict[str, Any]] = []
    tag = model_tag or Path(run_dir).name
    for study_id in sorted(pred_map):
        raw_rows = _read_prediction_csv(pred_map[study_id])
        prepared = prepare_ler_records(raw_rows)
        for rec in prepared:
            rec["model_tag"] = tag
            rec["method"] = method
            # Prefer study_id from filename when CSV metadata is blank.
            sid = rec.get("study_id")
            if sid is None or str(sid).strip() == "":
                rec["study_id"] = study_id
            else:
                try:
                    rec["study_id"] = int(float(sid))
                except (TypeError, ValueError):
                    rec["study_id"] = study_id
            rec["source_csv"] = str(pred_map[study_id])
            records.append(rec)
    return pd.DataFrame(records)


def _has_mapped_study(value: Any) -> bool:
    try:
        sid = int(float(value))
    except (TypeError, ValueError):
        return False
    return sid >= 1


def mark_regression_eligibility(prepared: pd.DataFrame) -> pd.DataFrame:
    """
    Flag rows eligible for matching / regression.

    Eligible means: final LER, final TND, and a mapped Study number.
    Matching is performed only on these rows.
    """
    out = prepared.copy()
    ler = pd.to_numeric(out.get("LER_final"), errors="coerce")
    tnd = pd.to_numeric(out.get("TND_final"), errors="coerce")
    study_ok = out.get("study_id", pd.Series(index=out.index)).map(_has_mapped_study)
    ler_ok = ler.notna() & out.get("LER_source", pd.Series(index=out.index)).ne("excluded")
    tnd_ok = tnd.notna() & out.get("TND_source", pd.Series(index=out.index)).ne("excluded")

    out["regression_eligible"] = (ler_ok & tnd_ok & study_ok).astype(bool)
    reasons: List[Optional[str]] = []
    for i in out.index:
        if bool(out.at[i, "regression_eligible"]):
            reasons.append(None)
            continue
        if not bool(study_ok.loc[i]):
            reasons.append("missing_mapped_study")
        elif not bool(ler_ok.loc[i]):
            reasons.append("missing_final_ler")
        elif not bool(tnd_ok.loc[i]):
            reasons.append("missing_final_tnd")
        else:
            reasons.append("not_regression_eligible")
    out["eligibility_exclusion_reason"] = reasons
    return out


def attach_experimental_ids(
    prepared: pd.DataFrame,
    gt_by_id: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    match_fields: Sequence[str] = MATCH_FIELDS,
) -> pd.DataFrame:
    """
    Match regression-eligible GenAI rows 1:1 to GT rows on design fields.

    - Only eligible rows (final LER + TND + Study) enter matching.
    - Each GT row matches at most one GenAI row.
    - Multiple GenAI rows may share the same GT ``Experimental ID`` when they
      match different GT rows from the same experiment.
    - Does not copy GT LER, TND, yields, or dates.
    """
    out = mark_regression_eligibility(prepared)
    out["Experimental ID"] = None
    out["matched_gt"] = False
    out["match_score"] = np.nan
    out["match_exclusion_reason"] = None
    out["matched_gt_row_index"] = pd.Series([pd.NA] * len(out), dtype="Int64")

    if out.empty:
        return out

    # Non-eligible rows are never matched.
    ineligible = ~out["regression_eligible"].astype(bool)
    out.loc[ineligible, "match_exclusion_reason"] = out.loc[
        ineligible, "eligibility_exclusion_reason"
    ]

    frames: List[pd.DataFrame] = []
    for study_id, group in out.groupby("study_id", sort=True):
        g = group.copy()
        try:
            sid = int(float(study_id))
        except (TypeError, ValueError):
            eligible_mask = g["regression_eligible"].astype(bool)
            g.loc[eligible_mask, "match_exclusion_reason"] = "missing_mapped_study"
            frames.append(g)
            continue

        gt_rows = list(gt_by_id.get(sid, []))
        eligible_idx = list(g.index[g["regression_eligible"].astype(bool)])
        if not eligible_idx:
            frames.append(g)
            continue
        if not gt_rows:
            g.loc[eligible_idx, "match_exclusion_reason"] = "no_gt_rows_for_study"
            frames.append(g)
            continue

        pred_rows = [g.loc[i].to_dict() for i in eligible_idx]
        usable = [
            f
            for f in match_fields
            if any(f in r for r in pred_rows) or any(f in r for r in gt_rows)
        ]
        matches = []
        if usable:
            matches = match_records_greedy_with_units(
                pred_rows, gt_rows, usable, gt_spreadsheet=True
            )

        matched_local = {pi: (gi, ev) for pi, gi, ev in matches}
        for local_i, row_i in enumerate(eligible_idx):
            if local_i in matched_local:
                gt_i, ev = matched_local[local_i]
                gt_row = gt_rows[gt_i]
                g.at[row_i, "Experimental ID"] = gt_row.get("Experimental ID")
                g.at[row_i, "matched_gt"] = True
                g.at[row_i, "match_score"] = float(ev.mean_score)
                g.at[row_i, "matched_gt_row_index"] = int(gt_i)
                g.at[row_i, "match_exclusion_reason"] = None
            else:
                g.at[row_i, "matched_gt"] = False
                g.at[row_i, "match_score"] = np.nan
                g.at[row_i, "match_exclusion_reason"] = "unmatched_eligible_row"
        frames.append(g)

    return pd.concat(frames, ignore_index=True) if frames else out


def analysis_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rows used in the nested LME.

    Requires regression eligibility (LER + TND + Study) and a successful
    design-field match that supplies ``Experimental ID`` as ``Expt``.
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    if "regression_eligible" not in out.columns:
        out = mark_regression_eligibility(out)
    out["Study"] = pd.to_numeric(out["study_id"], errors="coerce")
    out["Expt"] = out["Experimental ID"].astype(str)
    out["LER_final"] = pd.to_numeric(out["LER_final"], errors="coerce")
    out["TND_final"] = pd.to_numeric(out["TND_final"], errors="coerce")
    mask = (
        out["regression_eligible"].astype(bool)
        & out["matched_gt"].astype(bool)
        & out["Study"].notna()
        & out["Experimental ID"].notna()
        & out["Experimental ID"].astype(str).str.strip().ne("")
        & out["Experimental ID"].astype(str).str.lower().ne("nan")
        & out["Experimental ID"].astype(str).str.lower().ne("none")
    )
    return out.loc[mask].copy()


@dataclass
class LMEResult:
    model_tag: str
    method: str
    intercept: float
    tnd_coef: float
    tnd_se: float
    tnd_ci_low: float
    tnd_ci_high: float
    tnd_pvalue: float
    n_records: int
    n_studies: int
    n_experiments: int
    n_reported_ler: int
    n_calculated_ler: int
    n_excluded_ler: int
    n_prepared_rows: int
    engine: str
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_tag": self.model_tag,
            "method": self.method,
            "intercept": self.intercept,
            "tnd_coef": self.tnd_coef,
            "tnd_se": self.tnd_se,
            "tnd_ci_low": self.tnd_ci_low,
            "tnd_ci_high": self.tnd_ci_high,
            "tnd_pvalue": self.tnd_pvalue,
            "n_records": self.n_records,
            "n_studies": self.n_studies,
            "n_experiments": self.n_experiments,
            "n_reported_ler": self.n_reported_ler,
            "n_calculated_ler": self.n_calculated_ler,
            "n_excluded_ler": self.n_excluded_ler,
            "n_prepared_rows": self.n_prepared_rows,
            "engine": self.engine,
            "notes": self.notes,
        }


def _fit_lme_statsmodels(dat: pd.DataFrame) -> Tuple[float, float, float, float, float, float]:
    import statsmodels.formula.api as smf

    work = dat.copy()
    work["ExptNested"] = work["Study"].astype(int).astype(str) + ":" + work["Expt"].astype(str)
    md = smf.mixedlm(
        "LER_final ~ TND_final",
        work,
        groups=work["Study"],
        re_formula="1",
        vc_formula={"expt": "0 + C(ExptNested)"},
    )
    fit = md.fit(reml=True, method="lbfgs", maxiter=300)
    intercept = float(fit.params["Intercept"])
    coef = float(fit.params["TND_final"])
    se = float(fit.bse["TND_final"])
    ci = fit.conf_int().loc["TND_final"]
    pval = float(fit.pvalues["TND_final"])
    return intercept, coef, se, float(ci[0]), float(ci[1]), pval


def _fit_lme_r(
    dat: pd.DataFrame,
    r_script: Path,
) -> Tuple[float, float, float, float, float, float]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data_csv = tmp_path / "analysis_data.csv"
        out_json = tmp_path / "lme_result.json"
        export = dat[["Study", "Expt", "LER_final", "TND_final"]].copy()
        export.to_csv(data_csv, index=False)
        cmd = [
            "Rscript",
            str(r_script),
            "--data",
            str(data_csv),
            "--out",
            str(out_json),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"R nlme fit failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        return (
            float(payload["intercept"]),
            float(payload["tnd_coef"]),
            float(payload["tnd_se"]),
            float(payload["tnd_ci_low"]),
            float(payload["tnd_ci_high"]),
            float(payload["tnd_pvalue"]),
        )


def fit_ler_tnd_lme(
    analysis_df: pd.DataFrame,
    *,
    model_tag: str,
    method: str,
    prepared_df: Optional[pd.DataFrame] = None,
    prefer_r: bool = True,
    r_script: Optional[Path] = None,
) -> LMEResult:
    """Fit nested REML LME; prefer R ``nlme``, fall back to statsmodels."""
    prepared = prepared_df if prepared_df is not None else analysis_df
    n_reported = int((prepared.get("LER_source") == "reported").sum()) if len(prepared) else 0
    n_calc = (
        int((prepared.get("LER_source") == "calculated_from_yields").sum())
        if len(prepared)
        else 0
    )
    n_excl = int((prepared.get("LER_source") == "excluded").sum()) if len(prepared) else 0

    if analysis_df.empty or analysis_df["Study"].nunique() < 2:
        return LMEResult(
            model_tag=model_tag,
            method=method,
            intercept=float("nan"),
            tnd_coef=float("nan"),
            tnd_se=float("nan"),
            tnd_ci_low=float("nan"),
            tnd_ci_high=float("nan"),
            tnd_pvalue=float("nan"),
            n_records=int(len(analysis_df)),
            n_studies=int(analysis_df["Study"].nunique()) if len(analysis_df) else 0,
            n_experiments=0,
            n_reported_ler=n_reported,
            n_calculated_ler=n_calc,
            n_excluded_ler=n_excl,
            n_prepared_rows=int(len(prepared)),
            engine="none",
            notes="insufficient_rows_for_lme",
        )

    n_expts = (
        analysis_df.assign(
            _en=analysis_df["Study"].astype(int).astype(str)
            + ":"
            + analysis_df["Expt"].astype(str)
        )["_en"].nunique()
    )

    engine = "statsmodels"
    notes = ""
    r_path = r_script or (
        find_repo_root() / "notebooks" / "eval" / "fit_ler_tnd_lme.R"
    )
    intercept = coef = se = lo = hi = pval = float("nan")

    if prefer_r and r_path.is_file():
        try:
            intercept, coef, se, lo, hi, pval = _fit_lme_r(analysis_df, r_path)
            engine = "nlme"
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            notes = f"r_unavailable:{exc}"
            intercept, coef, se, lo, hi, pval = _fit_lme_statsmodels(analysis_df)
            engine = "statsmodels"
    else:
        intercept, coef, se, lo, hi, pval = _fit_lme_statsmodels(analysis_df)
        engine = "statsmodels"

    return LMEResult(
        model_tag=model_tag,
        method=method,
        intercept=intercept,
        tnd_coef=coef,
        tnd_se=se,
        tnd_ci_low=lo,
        tnd_ci_high=hi,
        tnd_pvalue=pval,
        n_records=int(len(analysis_df)),
        n_studies=int(analysis_df["Study"].nunique()),
        n_experiments=int(n_expts),
        n_reported_ler=n_reported,
        n_calculated_ler=n_calc,
        n_excluded_ler=n_excl,
        n_prepared_rows=int(len(prepared)),
        engine=engine,
        notes=notes,
    )


def ground_truth_baselines(
    gt_path: Optional[Path] = None,
) -> Dict[str, LMEResult]:
    """Published (1–100) and comparable (1–90) GT baselines."""
    path = gt_path or default_gt_path()
    gt = load_ground_truth(str(path))
    dat = pd.DataFrame(
        {
            "Study": pd.to_numeric(gt["Study#"], errors="coerce"),
            "Expt": gt["Experimental ID"].astype(str),
            "LER_final": pd.to_numeric(gt["LER"], errors="coerce"),
            "TND_final": pd.to_numeric(gt["TND"], errors="coerce"),
        }
    ).dropna(subset=["Study", "LER_final", "TND_final"])
    dat["Study"] = dat["Study"].astype(int)

    results: Dict[str, LMEResult] = {}
    for label, subset in (
        ("published_1_100", dat),
        ("comparable_1_90", dat.loc[dat["Study"] <= MAPPED_STUDY_MAX]),
    ):
        fit = fit_ler_tnd_lme(
            subset,
            model_tag=label,
            method="ground_truth",
            prepared_df=pd.DataFrame(
                {
                    "LER_source": ["reported"] * len(subset),
                }
            ),
            prefer_r=False,
        )
        results[label] = fit
    return results


def disagreement_report(prepared: pd.DataFrame) -> pd.DataFrame:
    """Rows where reported and yield-derived LER disagree beyond 2%."""
    if prepared.empty:
        return prepared.copy()
    mask = prepared["LER_agreement"] == "disagree"
    cols = [
        c
        for c in [
            "model_tag",
            "method",
            "study_id",
            "paper_id",
            "LER_reported_raw",
            "LER_reported_numeric",
            "LER_from_yields",
            "LER_relative_difference",
            "yield_sc1_raw",
            "yield_sc2_raw",
            "yield_ic1_raw",
            "yield_ic2_raw",
            "yield_unit_raw",
            "yield_unit_canonical",
            "yield_sc1_t_ha",
            "yield_sc2_t_ha",
            "yield_ic1_t_ha",
            "yield_ic2_t_ha",
            "source_csv",
        ]
        if c in prepared.columns
    ]
    return prepared.loc[mask, cols].copy()


def _draw_break_marks(ax_top, ax_bottom, *, d: float = 0.015) -> None:
    """Diagonal marks indicating a broken y-axis between two stacked axes."""
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, lw=1.0)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs = dict(transform=ax_bottom.transAxes, color="k", clip_on=False, lw=1.0)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)


def _plot_regression_overlays(
    ax,
    *,
    fit: LMEResult,
    baselines: Optional[Mapping[str, LMEResult]],
    x: np.ndarray,
    show_labels: bool,
) -> None:
    if math.isfinite(fit.tnd_coef) and math.isfinite(fit.intercept):
        y = fit.intercept + fit.tnd_coef * x
        ax.plot(
            x,
            y,
            color="#C44E52",
            lw=2.0,
            label=f"GenAI fit (β={fit.tnd_coef:.3f})" if show_labels else None,
        )
        if math.isfinite(fit.tnd_se):
            y_lo = fit.intercept + (fit.tnd_coef - 1.96 * fit.tnd_se) * x
            y_hi = fit.intercept + (fit.tnd_coef + 1.96 * fit.tnd_se) * x
            ax.fill_between(x, y_lo, y_hi, color="#C44E52", alpha=0.15, linewidth=0)

    if not baselines:
        return
    pub = baselines.get("published_1_100")
    cmp_ = baselines.get("comparable_1_90")
    if pub and math.isfinite(pub.tnd_coef):
        ax.plot(
            x,
            pub.intercept + pub.tnd_coef * x,
            color="#333333",
            ls="--",
            lw=1.5,
            label=f"Published GT β≈{pub.tnd_coef:.3f}" if show_labels else None,
        )
    if cmp_ and math.isfinite(cmp_.tnd_coef):
        ax.plot(
            x,
            cmp_.intercept + cmp_.tnd_coef * x,
            color="#55A868",
            ls=":",
            lw=1.8,
            label=f"GT studies 1–90 β={cmp_.tnd_coef:.3f}" if show_labels else None,
        )


def plot_figure8a_style(
    analysis_df: pd.DataFrame,
    fit: LMEResult,
    *,
    baselines: Optional[Mapping[str, LMEResult]] = None,
    ax=None,
    title: Optional[str] = None,
    ler_main_max: float = PLOT_LER_MAIN_MAX,
    ler_main_min: float = PLOT_LER_MAIN_MIN,
    ler_break_threshold: float = PLOT_LER_BREAK_THRESHOLD,
):
    """
    Scatter of TND vs LER with fixed-effect line and reference slopes.

    A broken / jumped y-axis is used only when some LER values are
    ``>= ler_break_threshold`` (default 10). Smaller outliers simply expand
    the single-axis y-limits.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    x = np.linspace(0.0, 1.0, 50)
    label = title or f"{fit.model_tag} · {fit.method}"
    subtitle = (
        f"n={fit.n_records} records · {fit.n_studies} studies · "
        f"{fit.n_experiments} expts · engine={fit.engine}"
    )

    def _single_axis(y_min: float, y_max: float, y_plot=None, x_plot=None):
        nonlocal ax
        if ax is None:
            _, ax = plt.subplots(figsize=(6.5, 5.0))
        if x_plot is not None and y_plot is not None and len(y_plot):
            ax.scatter(
                x_plot,
                y_plot,
                s=18,
                alpha=0.55,
                color="#4C72B0",
                edgecolors="none",
                label="GenAI records",
            )
        _plot_regression_overlays(ax, fit=fit, baselines=baselines, x=x, show_labels=True)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("TND")
        ax.set_ylabel("LER")
        ax.set_title(f"{label}\n{subtitle}")
        ax.legend(loc="best", fontsize=8, frameon=False)
        ax.grid(True, alpha=0.25)
        return ax

    if analysis_df is None or analysis_df.empty:
        return _single_axis(ler_main_min, ler_main_max)

    y_vals = pd.to_numeric(analysis_df["LER_final"], errors="coerce")
    x_vals = pd.to_numeric(analysis_df["TND_final"], errors="coerce")
    finite = y_vals.notna() & x_vals.notna()
    y_finite = y_vals[finite]
    x_finite = x_vals[finite]

    # Broken axis only for very large LER (>= threshold). Otherwise one panel.
    needs_break = bool((y_finite >= ler_break_threshold).any())
    if not needs_break:
        y_hi = float(y_finite.max()) if len(y_finite) else ler_main_max
        y_lo = float(y_finite.min()) if len(y_finite) else ler_main_min
        return _single_axis(
            min(ler_main_min, y_lo - 0.05),
            max(ler_main_max, y_hi + 0.1),
            y_plot=y_finite,
            x_plot=x_finite,
        )

    extreme_mask = y_finite >= ler_break_threshold
    main_mask = ~extreme_mask

    # Broken axis: upper panel for extreme LER, lower for the main range.
    fig = None
    if ax is None:
        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(6.5, 5.5),
            gridspec_kw={"height_ratios": [1, 3], "hspace": 0.08},
        )
    else:
        fig = ax.figure
        subplot_spec = ax.get_subplotspec()
        ax.remove()
        gs = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=subplot_spec, height_ratios=[1, 3], hspace=0.08
        )
        ax_top = fig.add_subplot(gs[0])
        ax_bottom = fig.add_subplot(gs[1], sharex=ax_top)

    ax_bottom.scatter(
        x_finite[main_mask],
        y_finite[main_mask],
        s=18,
        alpha=0.55,
        color="#4C72B0",
        edgecolors="none",
        label="GenAI records",
    )
    ax_top.scatter(
        x_finite[extreme_mask],
        y_finite[extreme_mask],
        s=22,
        alpha=0.75,
        color="#C44E52",
        edgecolors="none",
        label=f"LER ≥ {ler_break_threshold:g}",
    )

    _plot_regression_overlays(
        ax_bottom, fit=fit, baselines=baselines, x=x, show_labels=True
    )
    _plot_regression_overlays(
        ax_top, fit=fit, baselines=baselines, x=x, show_labels=False
    )

    y_hi = float(np.nanmax(y_finite[extreme_mask]))
    pad = max(0.5, 0.08 * (y_hi - ler_break_threshold))
    ax_top.set_ylim(ler_break_threshold, y_hi + pad)
    # Lower panel covers typical range up to just below the break.
    lower_hi = max(
        ler_main_max,
        float(y_finite[main_mask].max()) + 0.1 if main_mask.any() else ler_main_max,
    )
    # Keep a visual gap under the break threshold.
    lower_hi = min(lower_hi, ler_break_threshold * 0.95)
    ax_bottom.set_ylim(ler_main_min, max(lower_hi, ler_main_max))
    ax_top.set_xlim(-0.02, 1.02)
    ax_bottom.set_xlim(-0.02, 1.02)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_bottom.set_xlabel("TND")
    ax_bottom.set_ylabel("LER")
    ax_top.set_ylabel("LER")
    ax_top.set_title(f"{label}\n{subtitle}")
    _draw_break_marks(ax_top, ax_bottom)

    handles_b, labels_b = ax_bottom.get_legend_handles_labels()
    handles_t, labels_t = ax_top.get_legend_handles_labels()
    by_label = dict(zip(labels_b + labels_t, handles_b + handles_t))
    ax_bottom.legend(by_label.values(), by_label.keys(), loc="best", fontsize=8, frameon=False)
    ax_top.grid(True, alpha=0.25)
    ax_bottom.grid(True, alpha=0.25)
    return (ax_top, ax_bottom)


def run_model_method_pipeline(
    model_tag: str,
    method: str,
    *,
    outputs_root: Optional[Path] = None,
    processed_root: Optional[Path] = None,
    gt_path: Optional[Path] = None,
    baselines: Optional[Mapping[str, LMEResult]] = None,
    prefer_r: bool = True,
    write_outputs: bool = True,
    show_plot: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end reproduction for one model × method.

    Writes checkpoint CSVs under
    ``outputs/ler_tnd_reproduction/<model_tag>/<method>/`` without modifying
    original GenAI outputs.
    """
    root = find_repo_root()
    out_root = Path(outputs_root) if outputs_root else root / "outputs"
    run_dir = out_root / model_tag
    proc_root = Path(processed_root) if processed_root else default_processed_root(root)
    gt_file = Path(gt_path) if gt_path else default_gt_path(root)

    gt_by_id = load_ground_truth_by_study_id(str(gt_file))
    prepared = load_and_prepare_method(run_dir, method, model_tag=model_tag)
    prepared = attach_experimental_ids(prepared, gt_by_id)
    analysis = analysis_subset(prepared)
    fit = fit_ler_tnd_lme(
        analysis,
        model_tag=model_tag,
        method=method,
        prepared_df=prepared,
        prefer_r=prefer_r,
    )
    summary = exclusion_summary(
        prepared.to_dict(orient="records"),
        model_tag=model_tag,
        method=method,
    )
    summary["n_regression_eligible"] = int(prepared["regression_eligible"].sum()) if len(prepared) else 0
    summary["n_matched"] = int(prepared["matched_gt"].sum()) if len(prepared) else 0
    summary["n_analysis"] = int(len(analysis))
    if "match_exclusion_reason" in prepared.columns:
        match_excl = (
            prepared.loc[~prepared["matched_gt"].astype(bool), "match_exclusion_reason"]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .to_dict()
        )
        summary["match_exclusion_reasons"] = match_excl
    disagrees = disagreement_report(prepared)
    base = baselines or ground_truth_baselines(gt_file)

    dest = proc_root / model_tag / method
    if write_outputs:
        dest.mkdir(parents=True, exist_ok=True)
        prepared.to_csv(dest / "prepared_rows.csv", index=False)
        analysis.to_csv(dest / "analysis_rows.csv", index=False)
        disagrees.to_csv(dest / "ler_disagreements.csv", index=False)
        eligible = prepared.loc[prepared["regression_eligible"].astype(bool)].copy()
        eligible.to_csv(dest / "eligible_rows.csv", index=False)
        (dest / "exclusion_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        (dest / "lme_result.json").write_text(
            json.dumps(fit.as_dict(), indent=2), encoding="utf-8"
        )

    ax = None
    if show_plot:
        ax = plot_figure8a_style(analysis, fit, baselines=base)

    return {
        "prepared": prepared,
        "analysis": analysis,
        "fit": fit,
        "summary": summary,
        "disagreements": disagrees,
        "baselines": base,
        "output_dir": dest,
        "ax": ax,
    }


def run_all_methods_for_model(
    model_tag: str,
    *,
    methods: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """Run the pipeline for each extraction method of one model tag."""
    use_methods = list(methods) if methods is not None else list(METHOD_PREFIXES)
    # Share baselines across methods.
    baselines = kwargs.pop("baselines", None) or ground_truth_baselines(
        kwargs.get("gt_path")
    )
    results: Dict[str, Dict[str, Any]] = {}
    for method in use_methods:
        results[method] = run_model_method_pipeline(
            model_tag,
            method,
            baselines=baselines,
            **kwargs,
        )
    return results


__all__ = [
    "MATCH_FIELDS",
    "METHOD_PREFIXES",
    "PUBLISHED_TND_COEF",
    "MAPPED_STUDY_MAX",
    "PLOT_LER_MAIN_MAX",
    "PLOT_LER_MAIN_MIN",
    "PLOT_LER_BREAK_THRESHOLD",
    "LMEResult",
    "find_repo_root",
    "default_gt_path",
    "default_processed_root",
    "scan_by_paper",
    "list_model_tags",
    "load_and_prepare_method",
    "mark_regression_eligibility",
    "attach_experimental_ids",
    "analysis_subset",
    "fit_ler_tnd_lme",
    "ground_truth_baselines",
    "disagreement_report",
    "plot_figure8a_style",
    "run_model_method_pipeline",
    "run_all_methods_for_model",
    "FOLDER_TO_STUDY_ID",
]
