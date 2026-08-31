"""
Notebook helpers for evaluating wopke_100 extraction runs against ground truth.

Scoring, record matching, crop-swap, and FIELD/EXTRACTED/GT/SCORE printing all
come from ``src.experimentutils.eval_utils``. This module only discovers model
output folders, loads prediction CSVs, aggregates metrics, and plots.
"""
from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


METHOD_PREFIXES: Dict[str, str] = {
    "direct_llm": "direct_llm",
    "workflow": "static_workflow",
    "MAS": "mas",
}
PREFIX_TO_LABEL: Dict[str, str] = {v: k for k, v in METHOD_PREFIXES.items()}

METHOD_COLORS: Dict[str, str] = {
    "direct_llm": "#4C72B0",
    "workflow": "#55A868",
    "MAS": "#C44E52",
}

STACK_METRIC_KEYS = ("similarity", "factual_rate", "retrieval_rate")
OVERALL_SCORE_KEY = "overall_score"
METRIC_LABELS = {
    "similarity": "TP similarity",
    "factual_rate": "Factual rate (TP/(TP+FP))",
    "retrieval_rate": "Retrieval rate (TP/(TP+FN))",
    OVERALL_SCORE_KEY: "Overall score (true precision)",
}
METRIC_HATCHES = {
    "similarity": "",
    "factual_rate": "///",
    "retrieval_rate": "...",
}

_BY_PAPER_RE = re.compile(
    r"^(?P<sid>\d+)_.+__(?P<method>direct_llm|static_workflow|mas)\.csv$"
)
_DATED_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def find_repo_root(start: Optional[Path] = None) -> Path:
    p = start or Path.cwd()
    for candidate in [p, *p.parents]:
        if (candidate / "src").is_dir() and (candidate / "outputs").is_dir():
            return candidate
    return start or Path.cwd()


def ensure_src_on_path(repo_root: Optional[Path] = None) -> Path:
    root = repo_root or find_repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _safe01(v: float) -> float:
    return float(v) if v == v else 0.0


def _fmt(v: float) -> str:
    return f"{v:.3f}" if v == v else "nan"


def read_prediction_csv(csv_path: str) -> tuple[list[dict], list[str]]:
    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return rows, fieldnames


def list_model_runs(outputs_root: Optional[Path] = None) -> "pd.DataFrame":
    """Inventory ``outputs/<tag>/by_paper`` runs (one row per model tag)."""
    root = Path(outputs_root) if outputs_root else find_repo_root() / "outputs"
    rows: list[dict] = []
    if not root.exists():
        return pd.DataFrame(rows) if pd is not None else rows  # type: ignore

    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or _DATED_DIR_RE.match(run_dir.name):
            continue
        by_paper = run_dir / "by_paper"
        if not by_paper.is_dir():
            continue
        counts = {label: 0 for label in METHOD_PREFIXES}
        papers: set[int] = set()
        for csv_path in by_paper.glob("*.csv"):
            m = _BY_PAPER_RE.match(csv_path.name)
            if not m:
                continue
            label = PREFIX_TO_LABEL.get(m.group("method"))
            if label is None:
                continue
            counts[label] += 1
            papers.add(int(m.group("sid")))
        if not papers:
            continue
        rows.append(
            {
                "tag": run_dir.name,
                "run_dir": str(run_dir),
                "n_papers": len(papers),
                **{f"n_{k}": v for k, v in counts.items()},
            }
        )
    if pd is None:
        return rows  # type: ignore
    return pd.DataFrame(rows)


def scan_by_paper(run_dir: Path) -> Dict[str, Dict[int, str]]:
    """Return ``{method_label: {study_id: csv_path}}`` for a model run directory."""
    by_paper = Path(run_dir) / "by_paper"
    pred: Dict[str, Dict[int, str]] = {label: {} for label in METHOD_PREFIXES}
    if not by_paper.is_dir():
        raise FileNotFoundError(f"No by_paper folder in {run_dir}")
    for csv_path in by_paper.glob("*.csv"):
        m = _BY_PAPER_RE.match(csv_path.name)
        if not m:
            continue
        label = PREFIX_TO_LABEL.get(m.group("method"))
        if label is None:
            continue
        pred[label][int(m.group("sid"))] = str(csv_path)
    return pred


def load_gt(gt_path: Optional[Path] = None) -> tuple[Dict[int, List[Dict[str, Any]]], List[str]]:
    ensure_src_on_path()
    from src.experimentutils.eval_utils import (
        load_ground_truth_by_study_id,
        wopke_100_shared_fields,
    )

    path = gt_path
    if path is None:
        path = (
            find_repo_root()
            / "data"
            / "wopke_paper_code"
            / "Database for combined sample 2015-03-05.csv"
        )
    gt_by_id = load_ground_truth_by_study_id(str(path))
    some_rows = next(iter(gt_by_id.values()), [])
    gt_keys = some_rows[0].keys() if some_rows else []
    shared_fields = wopke_100_shared_fields(gt_keys)
    return gt_by_id, shared_fields


@dataclass
class ModelRun:
    tag: str
    run_dir: Path
    pred_by_method: Dict[str, Dict[int, str]]
    gt_by_id: Dict[int, List[Dict[str, Any]]]
    shared_fields: List[str]
    methods: List[str] = field(default_factory=lambda: list(METHOD_PREFIXES))

    @property
    def paper_ids(self) -> List[int]:
        found: set[int] = set()
        for m in self.methods:
            found.update(self.pred_by_method.get(m, {}))
        return sorted(found)

    def csv_path(self, paper_id: int, method: str) -> Optional[str]:
        return self.pred_by_method.get(method, {}).get(paper_id)

    def pred_rows(self, paper_id: int, method: str) -> List[Dict[str, Any]]:
        path = self.csv_path(paper_id, method)
        if path is None:
            return []
        rows, _ = read_prediction_csv(path)
        return rows


def load_model_run(
    tag: str,
    *,
    outputs_root: Optional[Path] = None,
    gt_path: Optional[Path] = None,
    methods: Optional[Sequence[str]] = None,
) -> ModelRun:
    root = Path(outputs_root) if outputs_root else find_repo_root() / "outputs"
    run_dir = root / tag
    if not run_dir.is_dir():
        available = sorted(p.name for p in root.iterdir() if (p / "by_paper").is_dir())
        raise FileNotFoundError(
            f"Model run not found: {run_dir}\nAvailable tags: {available}"
        )
    gt_by_id, shared_fields = load_gt(gt_path)
    pred = scan_by_paper(run_dir)
    use_methods = list(methods) if methods else [
        m for m in METHOD_PREFIXES if pred.get(m)
    ]
    if not use_methods:
        use_methods = list(METHOD_PREFIXES)
    return ModelRun(
        tag=tag,
        run_dir=run_dir,
        pred_by_method=pred,
        gt_by_id=gt_by_id,
        shared_fields=shared_fields,
        methods=use_methods,
    )


def paper_label(run: ModelRun, paper_id: int) -> str:
    rows = run.gt_by_id.get(paper_id, [])
    if not rows:
        return str(paper_id)
    rec = rows[0]
    author = rec.get("Author")
    title = rec.get("Title")
    a = author.strip() if isinstance(author, str) else ""
    t = title.strip() if isinstance(title, str) else ""
    if a and t:
        return f"{a} — {t}"
    return a or t or str(paper_id)


def paper_index_label(run: ModelRun, paper_id: int) -> str:
    rows = run.gt_by_id.get(paper_id, [])
    if not rows:
        return str(paper_id)
    author = rows[0].get("Author")
    year = rows[0].get("Year of publication")
    a = ""
    if isinstance(author, str) and author.strip():
        a = author.strip().split(";")[0].strip()
    y = ""
    if year is not None:
        try:
            y = str(int(float(year)))
        except (TypeError, ValueError):
            y = str(year).strip()
    if a and y:
        return f"{a}\n{y}"
    return a or y or str(paper_id)


def _progress(iterable, *, desc: str = "", show: bool = True):
    if not show:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc)
    except Exception:
        return iterable


def collect_greedy_matches(
    run: ModelRun,
    *,
    paper_ids: Optional[Sequence[int]] = None,
    methods: Optional[Sequence[str]] = None,
    fields: Optional[List[str]] = None,
    show_progress: bool = True,
) -> Dict[tuple[str, int], list]:
    """
    Greedy row matching once per (method, paper_id).

    Reused by paper-level metrics and per-field aggregates so we do not
    re-run O(n_pred × n_gt) matching for every field.
    """
    ensure_src_on_path()
    from src.experimentutils.eval_utils import (
        infer_gt_spreadsheet_columns,
        match_records_greedy_with_units,
    )

    use_fields = fields or run.shared_fields
    use_methods = list(methods) if methods else run.methods
    pids = list(paper_ids) if paper_ids is not None else run.paper_ids
    gt_ss = infer_gt_spreadsheet_columns(use_fields)

    tasks = [
        (m, pid)
        for pid in pids
        for m in use_methods
        if pid in run.gt_by_id and run.pred_rows(pid, m)
    ]
    out: Dict[tuple[str, int], list] = {}
    for m, pid in _progress(tasks, desc="Matching records", show=show_progress):
        pred_rows = run.pred_rows(pid, m)
        gt_rows = run.gt_by_id[pid]
        out[(m, pid)] = match_records_greedy_with_units(
            pred_rows, gt_rows, use_fields, gt_spreadsheet=gt_ss
        )
    return out


def paper_metrics(
    run: ModelRun,
    paper_id: int,
    method: str,
    *,
    fields: Optional[List[str]] = None,
) -> Dict[str, float]:
    ensure_src_on_path()
    from src.experimentutils.eval_utils import (
        infer_gt_spreadsheet_columns,
        overall_metrics_with_units,
    )

    gt_rows = run.gt_by_id.get(paper_id, [])
    if not gt_rows:
        raise KeyError(f"paper_id={paper_id} not found in GT")
    use_fields = fields or run.shared_fields
    pred_rows = run.pred_rows(paper_id, method)
    met = overall_metrics_with_units(
        pred_rows,
        gt_rows,
        use_fields,
        gt_spreadsheet=infer_gt_spreadsheet_columns(use_fields),
    )
    return {
        "similarity": _safe01(met.get("tp_avg_similarity", float("nan"))),
        "factual_rate": _safe01(met.get("precision", float("nan"))),
        "retrieval_rate": _safe01(met.get("recall", float("nan"))),
        OVERALL_SCORE_KEY: _safe01(met.get("true_precision", float("nan"))),
        "f1": _safe01(met.get("f1", float("nan"))),
        "TP": met.get("TP", 0),
        "FP": met.get("FP", 0),
        "FN": met.get("FN", 0),
        "TN": met.get("TN", 0),
        "n_pred": len(pred_rows),
        "n_gt": len(gt_rows),
        **met,
    }


def all_papers_metrics_df(
    run: ModelRun,
    *,
    paper_ids: Optional[Sequence[int]] = None,
    methods: Optional[Sequence[str]] = None,
    fields: Optional[List[str]] = None,
    show_progress: bool = True,
) -> "pd.DataFrame":
    ensure_src_on_path()
    from src.experimentutils.eval_utils import (
        infer_gt_spreadsheet_columns,
        overall_metrics_with_units,
    )

    use_methods = list(methods) if methods else run.methods
    pids = list(paper_ids) if paper_ids is not None else run.paper_ids
    use_fields = fields or run.shared_fields
    gt_ss = infer_gt_spreadsheet_columns(use_fields)

    rows: list[dict] = []
    tasks = [(pid, m) for pid in pids if pid in run.gt_by_id for m in use_methods]
    for pid, m in _progress(tasks, desc="Scoring papers", show=show_progress):
        gt_rows = run.gt_by_id[pid]
        pred_rows = run.pred_rows(pid, m)
        met = overall_metrics_with_units(
            pred_rows,
            gt_rows,
            use_fields,
            gt_spreadsheet=gt_ss,
        )
        rows.append(
            {
                "paper_id": pid,
                "method": m,
                "paper": paper_index_label(run, pid),
                "n_pred": len(pred_rows),
                "n_gt": len(gt_rows),
                "similarity": _safe01(met.get("tp_avg_similarity", float("nan"))),
                "factual_rate": _safe01(met.get("precision", float("nan"))),
                "retrieval_rate": _safe01(met.get("recall", float("nan"))),
                OVERALL_SCORE_KEY: _safe01(met.get("true_precision", float("nan"))),
                "f1": _safe01(met.get("f1", float("nan"))),
            }
        )
    if pd is None:
        raise RuntimeError("pandas is required for all_papers_metrics_df")
    return pd.DataFrame(rows)


def overall_mean_metrics_df(metrics_df: "pd.DataFrame") -> "pd.DataFrame":
    methods = list(dict.fromkeys(metrics_df["method"].tolist()))
    rows = []
    for m in methods:
        sub = metrics_df[metrics_df["method"] == m]
        row = {
            "method": m,
            "n_papers": int(sub["paper_id"].nunique()),
        }
        for k in (*STACK_METRIC_KEYS, OVERALL_SCORE_KEY, "f1"):
            row[k] = float(sub[k].mean()) if len(sub) else float("nan")
        row["stacked_total"] = sum(row[k] for k in STACK_METRIC_KEYS)
        rows.append(row)
    return pd.DataFrame(rows)


def print_model_scores(mean_df: "pd.DataFrame", *, tag: str = "") -> None:
    title = f"Overall mean scores — {tag}" if tag else "Overall mean scores"
    print(title)
    print("-" * len(title))
    for _, row in mean_df.iterrows():
        print(
            f"  {row['method']:<12}  papers={int(row['n_papers']):3d}  "
            f"overall={_fmt(row[OVERALL_SCORE_KEY])}  "
            f"TP-sim={_fmt(row['similarity'])}  "
            f"factual={_fmt(row['factual_rate'])}  "
            f"retrieval={_fmt(row['retrieval_rate'])}  "
            f"f1={_fmt(row['f1'])}"
        )


def _draw_stack(ax, x_pos: float, heights: dict[str, float], color: str, bar_width: float, *, alpha: float = 1.0):
    bottom = 0.0
    for key in STACK_METRIC_KEYS:
        ax.bar(
            x_pos,
            heights[key],
            bar_width,
            bottom=bottom,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            hatch=METRIC_HATCHES[key],
            alpha=alpha,
        )
        bottom += heights[key]


def plot_overall_stacked(mean_df: "pd.DataFrame", *, title: str = "") -> None:
    methods = list(mean_df["method"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    x = np.arange(len(methods))
    for i, m in enumerate(methods):
        row = mean_df[mean_df["method"] == m].iloc[0]
        stack_h = {k: float(row[k]) for k in STACK_METRIC_KEYS}
        _draw_stack(ax, x[i], stack_h, METHOD_COLORS.get(m, "steelblue"), 0.55)
        ax.text(
            x[i],
            float(row["stacked_total"]) + 0.06,
            f"overall={_fmt(row[OVERALL_SCORE_KEY])}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 3.4)
    ax.set_ylabel("Stacked score (each segment ∈ [0, 1])")
    ax.set_title(title or "Overall mean metrics by method")
    for y in (1.0, 2.0, 3.0):
        ax.axhline(y, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    stack_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=METRIC_HATCHES[k], label=METRIC_LABELS[k])
        for k in STACK_METRIC_KEYS
    ]
    ax.legend(handles=stack_handles, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    plt.tight_layout()
    plt.show()


def plot_paper_stacked_bars(
    run: ModelRun,
    metrics_df: "pd.DataFrame",
    mean_df: "pd.DataFrame",
    *,
    bar_width: float = 0.22,
) -> None:
    methods = [m for m in run.methods if m in set(metrics_df["method"])]
    pids = sorted(metrics_df["paper_id"].unique())
    n_papers = len(pids)
    n_methods = len(methods)
    overall_mean_x = n_papers
    n_groups = n_papers + 1
    x = np.arange(n_groups)
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * bar_width
    figsize = (min(max(12.0, n_groups * 0.55), 36.0), 6.2)

    fig, ax = plt.subplots(figsize=figsize)
    for mi, m in enumerate(methods):
        sub = metrics_df[metrics_df["method"] == m].set_index("paper_id")
        color = METHOD_COLORS.get(m, "steelblue")
        for pi, pid in enumerate(pids):
            if pid not in sub.index:
                continue
            stack_h = {k: float(sub.loc[pid, k]) for k in STACK_METRIC_KEYS}
            _draw_stack(ax, x[pi] + offsets[mi], stack_h, color, bar_width)
        mean_row = mean_df[mean_df["method"] == m].iloc[0]
        stack_mean = {k: float(mean_row[k]) for k in STACK_METRIC_KEYS}
        _draw_stack(ax, overall_mean_x + offsets[mi], stack_mean, color, bar_width, alpha=0.55)

    labels = [paper_index_label(run, pid) for pid in pids] + ["Overall\nmean"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=7)
    ax.axvline(overall_mean_x - 0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_ylabel("Score (each segment ∈ [0, 1])")
    ax.set_ylim(0, 3.25)
    ax.set_title(f"Paper metrics by method — {run.tag}", fontsize=11)
    for y in (1.0, 2.0, 3.0):
        ax.axhline(y, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    stack_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=METRIC_HATCHES[k], label=METRIC_LABELS[k])
        for k in STACK_METRIC_KEYS
    ]
    method_handles = [
        Patch(facecolor=METHOD_COLORS.get(m, "steelblue"), edgecolor="black", label=m)
        for m in methods
    ]
    leg1 = ax.legend(handles=stack_handles, title="Metric", loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=method_handles, title="Method", loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    plt.tight_layout()
    plt.show()


def per_field_avg_across_papers_df(
    run: ModelRun,
    *,
    paper_ids: Optional[Sequence[int]] = None,
    methods: Optional[Sequence[str]] = None,
    fields: Optional[List[str]] = None,
    show_progress: bool = True,
    match_cache: Optional[Dict[tuple[str, int], list]] = None,
) -> "pd.DataFrame":
    use_fields = fields or run.shared_fields
    use_methods = list(methods) if methods else run.methods

    if match_cache is None:
        match_cache = collect_greedy_matches(
            run,
            paper_ids=paper_ids,
            methods=use_methods,
            fields=use_fields,
            show_progress=show_progress,
        )

    field_avgs: Dict[str, Dict[str, list[float]]] = {
        f: {m: [] for m in use_methods} for f in use_fields
    }
    for (m, _pid), matches in match_cache.items():
        for f in use_fields:
            scores = [ev.field_scores.get(f, 0.0) for _, _, ev in matches]
            if scores:
                field_avgs[f][m].append(float(np.mean(scores)))

    rows: list[dict] = []
    for f in use_fields:
        row: dict = {"field": f}
        for m in use_methods:
            avgs = field_avgs[f][m]
            row[m] = float(np.mean(avgs)) if avgs else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def plot_per_field_avg(
    field_df: "pd.DataFrame",
    *,
    methods: Sequence[str],
    title: str = "",
    include_overall_mean_column: bool = True,
) -> "pd.DataFrame":
    df = field_df.copy()
    use_methods = list(methods)
    if include_overall_mean_column:
        overall_row = {"field": "Overall mean"}
        for m in use_methods:
            overall_row[m] = float(df[m].mean())
        df = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)

    n_groups = len(df)
    figsize = (max(14.0, n_groups * 0.38), 6.5)
    x = np.arange(n_groups)
    width = 0.8 / max(len(use_methods), 1)

    fig, ax = plt.subplots(figsize=figsize)
    for i, m in enumerate(use_methods):
        offset = (i - (len(use_methods) - 1) / 2) * width
        ax.bar(
            x + offset,
            df[m].values,
            width=width,
            label=m,
            color=METHOD_COLORS.get(m, "steelblue"),
            alpha=0.9,
        )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean field score (avg across papers)")
    ax.set_xlabel("Field")
    ax.set_title(title or "Per-field scores by method (mean across papers)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(df["field"], rotation=90, ha="center", fontsize=7)
    if include_overall_mean_column and n_groups > 1:
        ax.axvline(n_groups - 1.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.legend(title="Method", loc="upper right")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    plt.tight_layout()
    plt.show()
    return df


def eval_model(
    run: ModelRun,
    *,
    paper_ids: Optional[Sequence[int]] = None,
    plot_per_paper: bool = True,
    plot_fields: bool = True,
    show_progress: bool = True,
    max_papers_in_plot: int = 30,
) -> dict[str, Any]:
    """Score one model run, print overall numbers, and draw plots."""
    metrics_df = all_papers_metrics_df(
        run, paper_ids=paper_ids, show_progress=show_progress
    )
    if metrics_df.empty:
        print(f"No scored papers for {run.tag}")
        return {"metrics": metrics_df, "mean": None, "fields": None}

    mean_df = overall_mean_metrics_df(metrics_df)
    print_model_scores(mean_df, tag=run.tag)
    print()
    plot_overall_stacked(mean_df, title=f"Overall mean — {run.tag}")

    if plot_per_paper:
        plot_pids = sorted(metrics_df["paper_id"].unique())
        if len(plot_pids) > max_papers_in_plot:
            print(
                f"Per-paper plot: showing first {max_papers_in_plot} of "
                f"{len(plot_pids)} papers (set max_papers_in_plot to change)."
            )
            plot_pids = plot_pids[:max_papers_in_plot]
            plot_metrics = metrics_df[metrics_df["paper_id"].isin(plot_pids)]
        else:
            plot_metrics = metrics_df
        plot_paper_stacked_bars(run, plot_metrics, mean_df)

    field_df = None
    match_cache = None
    if plot_fields:
        match_cache = collect_greedy_matches(
            run, paper_ids=paper_ids, show_progress=show_progress
        )
        field_df = per_field_avg_across_papers_df(
            run,
            paper_ids=paper_ids,
            show_progress=show_progress,
            match_cache=match_cache,
        )
        plot_per_field_avg(
            field_df,
            methods=run.methods,
            title=f"Per-field mean scores — {run.tag}",
        )

    return {"metrics": metrics_df, "mean": mean_df, "fields": field_df}


def print_paper_vs_gt(
    run: ModelRun,
    paper_id: int,
    method: Optional[str] = None,
    *,
    max_pairs: Optional[int] = None,
    show_fields: Optional[List[str]] = None,
) -> None:
    """
    Print greedy-matched records for one paper vs ground truth.

    Same column layout as ``print_matching_pairs_from_df``:
    FIELD / EXTRACTED / GT / SCORE.
    """
    ensure_src_on_path()
    from src.experimentutils.eval_utils import (
        infer_gt_spreadsheet_columns,
        overall_metrics_with_units,
        print_matching_pairs_from_df,
    )

    gt_rows = run.gt_by_id.get(paper_id)
    if not gt_rows:
        raise KeyError(f"paper_id={paper_id} not found in GT")

    methods = [method] if method else run.methods
    unknown = [m for m in methods if m not in METHOD_PREFIXES]
    if unknown:
        raise ValueError(f"Unknown method(s) {unknown}; choose from {list(METHOD_PREFIXES)}")

    gt_df = pd.DataFrame(gt_rows)
    fields = run.shared_fields
    gt_ss = infer_gt_spreadsheet_columns(fields)

    print(f"model     : {run.tag}")
    print(f"paper_id  : {paper_id} | {paper_label(run, paper_id)}")
    print(f"GT rows   : {len(gt_rows)}")
    print(f"fields    : {len(fields)}")
    print()

    for m in methods:
        path = run.csv_path(paper_id, m)
        pred_rows = run.pred_rows(paper_id, m)
        print("=" * 88)
        print(f"method={m}")
        print(f"csv   : {path or '[missing]'}")
        if path is None:
            print("  [no prediction CSV for this paper]")
            print()
            continue

        met = overall_metrics_with_units(
            pred_rows, gt_rows, fields, gt_spreadsheet=gt_ss
        )
        print(
            f"  score  overall={_fmt(met.get('true_precision', float('nan')))}  "
            f"TP-sim={_fmt(met.get('tp_avg_similarity', float('nan')))}  "
            f"factual={_fmt(met.get('precision', float('nan')))}  "
            f"retrieval={_fmt(met.get('recall', float('nan')))}  "
            f"f1={_fmt(met.get('f1', float('nan')))}"
        )
        print()
        ext_df = pd.DataFrame(pred_rows)
        print_matching_pairs_from_df(
            ext_df,
            gt_df,
            fields,
            max_pairs=max_pairs,
            show_fields=show_fields,
        )
