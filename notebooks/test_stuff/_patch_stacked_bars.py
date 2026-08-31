"""One-off patch: two-panel plot for paper_metrics_stacked_bars.ipynb"""
import json
from pathlib import Path

path = Path(__file__).with_name("paper_metrics_stacked_bars.ipynb")
nb = json.loads(path.read_text())

new_plot = Path(__file__).with_name("_plot_cell.py").read_text() if False else None

# inlined below
NEW_PLOT = r'''import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

METHOD_COLORS = {"direct_llm": "#4C72B0", "workflow": "#55A868", "MAS": "#C44E52"}

STACK_METRIC_KEYS = ("similarity", "factual_rate", "retrieval_rate")
OVERALL_SCORE_KEY = "overall_score"
METRIC_KEYS = STACK_METRIC_KEYS + (OVERALL_SCORE_KEY,)

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
    OVERALL_SCORE_KEY: "xx",
}


def _safe01(v: float) -> float:
    return float(v) if v == v else 0.0


def paper_three_metrics(
    paper_id: int,
    method: str,
    *,
    fields: list[str] | None = None,
) -> dict[str, float]:
    gt_rows = gt_by_id.get(paper_id, [])
    if not gt_rows:
        raise KeyError(f"paper_id={paper_id} not found in GT")

    use_fields = fields or shared_fields
    csv_path = pred_by_method.get(method, {}).get(paper_id)
    pred_rows = []
    if csv_path is not None:
        pred_rows, _ = read_prediction_csv(csv_path)

    met = overall_metrics(pred_rows, gt_rows, use_fields)
    return {
        "similarity": _safe01(met.get("tp_avg_similarity", float("nan"))),
        "factual_rate": _safe01(met.get("precision", float("nan"))),
        "retrieval_rate": _safe01(met.get("recall", float("nan"))),
        OVERALL_SCORE_KEY: _safe01(met.get("true_precision", float("nan"))),
    }


def all_papers_metrics_df(
    paper_ids: list[int] | None = None,
    *,
    methods: list[str] | None = None,
    fields: list[str] | None = None,
) -> "pd.DataFrame":
    use_methods = methods or ["direct_llm", "workflow", "MAS"]
    pids = paper_ids or sorted(
        set().union(*(set(pred_by_method[m].keys()) for m in METHODS.keys()))
    )

    rows: list[dict] = []
    for pid in pids:
        for m in use_methods:
            scores = paper_three_metrics(pid, m, fields=fields)
            rows.append({"paper_id": pid, "method": m, **scores, "paper": paper_index_label(pid)})
    return pd.DataFrame(rows)


def overall_mean_metrics_df(
    paper_ids: list[int] | None = None,
    *,
    methods: list[str] | None = None,
    fields: list[str] | None = None,
) -> "pd.DataFrame":
    df = all_papers_metrics_df(paper_ids, methods=methods, fields=fields)
    use_methods = methods or ["direct_llm", "workflow", "MAS"]
    rows = []
    for m in use_methods:
        sub = df[df["method"] == m]
        row = {"method": m}
        for k in STACK_METRIC_KEYS:
            row[k] = float(sub[k].mean())
        row[OVERALL_SCORE_KEY] = float(sub[OVERALL_SCORE_KEY].mean())
        row["stacked_total"] = sum(row[k] for k in STACK_METRIC_KEYS)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_paper_metrics_stacked_bars(
    paper_ids: list[int] | None = None,
    *,
    methods: list[str] | None = None,
    fields: list[str] | None = None,
    bar_width: float = 0.22,
):
    """Two panels: per-paper stacks (top) + overall score & overall mean (bottom)."""
    use_methods = methods or ["direct_llm", "workflow", "MAS"]
    df = all_papers_metrics_df(paper_ids, methods=use_methods, fields=fields)
    pids = sorted(df["paper_id"].unique())
    mean_df = overall_mean_metrics_df(paper_ids, methods=use_methods, fields=fields)

    n_methods = len(use_methods)
    n_papers = len(pids)
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * bar_width

    print(f"TOP panel: {n_papers} paper groups  |  BOTTOM panel: 2 summary groups")

    fig, (ax_papers, ax_summary) = plt.subplots(
        2,
        1,
        figsize=(max(14, n_papers * 1.25), 9),
        gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.38},
    )

    def _draw_stack(ax, x_pos, heights, color, *, alpha=1.0):
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

    def _draw_overall_score_bar(ax, x_pos, height, color, *, alpha=1.0):
        ax.bar(
            x_pos,
            height,
            bar_width * 1.1,
            color=color,
            edgecolor="black",
            linewidth=0.9,
            hatch=METRIC_HATCHES[OVERALL_SCORE_KEY],
            alpha=alpha,
        )

    x_papers = np.arange(n_papers)
    for mi, m in enumerate(use_methods):
        sub = df[df["method"] == m].set_index("paper_id").loc[pids]
        color = METHOD_COLORS.get(m, "steelblue")
        for pi, pid in enumerate(pids):
            stack_h = {k: float(sub.loc[pid, k]) for k in STACK_METRIC_KEYS}
            _draw_stack(ax_papers, x_papers[pi] + offsets[mi], stack_h, color)

    ax_papers.set_xticks(x_papers)
    ax_papers.set_xticklabels([paper_index_label(pid) for pid in pids], fontsize=8)
    ax_papers.set_ylabel("Score")
    ax_papers.set_ylim(0, 3.15)
    ax_papers.set_title(f"Per-paper metrics ({n_papers} papers, 3 methods each)")
    ax_papers.grid(axis="y", linestyle=":", alpha=0.35)

    x_summary = np.array([0.0, 1.0])
    for mi, m in enumerate(use_methods):
        mean_row = mean_df[mean_df["method"] == m].iloc[0]
        color = METHOD_COLORS.get(m, "steelblue")
        _draw_overall_score_bar(
            ax_summary, x_summary[0] + offsets[mi], float(mean_row[OVERALL_SCORE_KEY]), color
        )
        stack_mean = {k: float(mean_row[k]) for k in STACK_METRIC_KEYS}
        _draw_stack(ax_summary, x_summary[1] + offsets[mi], stack_mean, color, alpha=0.85)

    ax_summary.set_xticks(x_summary)
    ax_summary.set_xticklabels(["Overall score\n(mean)", "Overall mean\n(stacked)"], fontweight="bold")
    ax_summary.set_ylabel("Score")
    ax_summary.set_ylim(0, 3.15)
    ax_summary.set_xlim(-0.6, 1.6)
    ax_summary.set_title("Summary across all papers")
    ax_summary.axvline(0.5, color="gray", linestyle="--", linewidth=1.0)
    ax_summary.grid(axis="y", linestyle=":", alpha=0.35)

    handles = (
        [Patch(facecolor="white", edgecolor="black", hatch=METRIC_HATCHES[k], label=METRIC_LABELS[k]) for k in STACK_METRIC_KEYS]
        + [Patch(facecolor="white", edgecolor="black", hatch=METRIC_HATCHES[OVERALL_SCORE_KEY], label=METRIC_LABELS[OVERALL_SCORE_KEY])]
        + [Patch(facecolor=METHOD_COLORS[m], edgecolor="black", label=m) for m in use_methods]
    )
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=9)
    plt.tight_layout()
    plt.show()
    return df, mean_df


paper_ids = sorted(set().union(*(set(pred_by_method[m].keys()) for m in METHODS.keys())))
metrics_df, mean_df = plot_paper_metrics_stacked_bars(paper_ids)
print(mean_df.to_string(index=False))
mean_df
'''

nb["cells"][4]["source"] = NEW_PLOT.splitlines(keepends=True)
nb["cells"][0]["source"] = [
    "# Paper-level stacked metrics\n\n"
    "**Restart kernel and Run All.**\n\n"
    "- **Top panel:** per-paper stacked bars (10 papers)\n"
    "- **Bottom panel:** 2 groups — **Overall score** (single bars) + **Overall mean** (stacked)\n"
]
for c in nb["cells"]:
    c["outputs"] = []
    c["execution_count"] = None

path.write_text(json.dumps(nb, indent=1))
print("patched", path)
