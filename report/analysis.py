"""Generate tables, charts, and statistical tests from benchmark results.

Usage:
    python -m report.analysis [--results benchmark/results/results.jsonl]
                               [--output report/figures]
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLORS = {
    "plain_english": "#4C72B0",
    "promptscript": "#DD8452",
}
TASK_TYPE_ORDER = ["factual_qa", "multi_hop", "summarization", "out_of_context"]
METRIC_LABELS = {
    "answer_correctness": "Answer Correctness",
    "faithfulness": "Faithfulness",
    "format_compliance": "Format Compliance",
    "token_efficiency": "Token Efficiency",
}
METRICS = list(METRIC_LABELS.keys())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_file: str | Path) -> pd.DataFrame:
    rows = []
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    # Normalise task_type values
    df["task_type"] = df["task_type"].str.replace("-", "_")
    return df


# ---------------------------------------------------------------------------
# Figure 1: Grouped bar chart — mean score per metric per prompt type
# ---------------------------------------------------------------------------

def fig_overall_bars(df: pd.DataFrame, out_dir: Path) -> None:
    means = df.groupby("prompt_type")[METRICS].mean()

    x = np.arange(len(METRICS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, pt in enumerate(["plain_english", "promptscript"]):
        if pt not in means.index:
            continue
        vals = [means.loc[pt, m] for m in METRICS]
        bars = ax.bar(x + i * width - width / 2, vals, width,
                      label=pt.replace("_", " ").title(),
                      color=COLORS[pt], alpha=0.88, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=10)
    ax.set_ylabel("Mean Score  [0 – 1]", fontsize=11)
    ax.set_title("Overall Mean Scores: Plain English vs PromptScript", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    path = out_dir / "fig1_overall_bars.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure 2: Grouped bar chart — correctness by task type
# ---------------------------------------------------------------------------

def fig_correctness_by_type(df: pd.DataFrame, out_dir: Path) -> None:
    present_types = [t for t in TASK_TYPE_ORDER if t in df["task_type"].unique()]
    pivot = (
        df.groupby(["task_type", "prompt_type"])["answer_correctness"]
        .mean()
        .unstack("prompt_type")
    )

    x = np.arange(len(present_types))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, pt in enumerate(["plain_english", "promptscript"]):
        if pt not in pivot.columns:
            continue
        vals = [pivot.loc[t, pt] if t in pivot.index else 0 for t in present_types]
        bars = ax.bar(x + i * width - width / 2, vals, width,
                      label=pt.replace("_", " ").title(),
                      color=COLORS[pt], alpha=0.88, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", " ").title() for t in present_types], fontsize=10)
    ax.set_ylabel("Mean Answer Correctness", fontsize=11)
    ax.set_title("Answer Correctness by Task Type", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    path = out_dir / "fig2_correctness_by_type.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure 3: Scatter — correctness vs faithfulness (per prompt type)
# ---------------------------------------------------------------------------

def fig_scatter_correctness_faithfulness(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    for pt in ["plain_english", "promptscript"]:
        sub = df[df["prompt_type"] == pt]
        if sub.empty:
            continue
        ax.scatter(
            sub["answer_correctness"], sub["faithfulness"],
            label=pt.replace("_", " ").title(),
            color=COLORS[pt], alpha=0.65, s=50, edgecolors="none"
        )

    ax.set_xlabel("Answer Correctness", fontsize=11)
    ax.set_ylabel("Faithfulness", fontsize=11)
    ax.set_title("Correctness vs Faithfulness (per task)", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)

    path = out_dir / "fig3_scatter_corr_faith.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure 4: Prompt token histogram
# ---------------------------------------------------------------------------

def fig_token_histogram(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for pt in ["plain_english", "promptscript"]:
        sub = df[df["prompt_type"] == pt]["prompt_tokens"].dropna()
        if sub.empty:
            continue
        ax.hist(sub, bins=20, alpha=0.65,
                label=pt.replace("_", " ").title(),
                color=COLORS[pt], edgecolor="white")

    ax.set_xlabel("Prompt Token Count", fontsize=11)
    ax.set_ylabel("Number of Tasks", fontsize=11)
    ax.set_title("Distribution of Prompt Token Counts", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    path = out_dir / "fig4_token_histogram.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure 5: Heatmap — mean score by task_type × metric
# ---------------------------------------------------------------------------

def fig_heatmap_by_type(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, pt in zip(axes, ["plain_english", "promptscript"]):
        sub = df[df["prompt_type"] == pt]
        pivot = sub.groupby("task_type")[METRICS].mean()
        # Reindex to canonical order
        pivot = pivot.reindex([t for t in TASK_TYPE_ORDER if t in pivot.index])
        pivot.index = [i.replace("_", " ").title() for i in pivot.index]
        pivot.columns = [METRIC_LABELS[c] for c in pivot.columns]

        im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=10)
        ax.set_title(pt.replace("_", " ").title(), fontsize=12, fontweight="bold")

        for r in range(len(pivot.index)):
            for c in range(len(pivot.columns)):
                val = pivot.values[r, c]
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color="black" if 0.3 < val < 0.8 else "white")

    fig.colorbar(im, ax=axes, label="Score", fraction=0.02, pad=0.04)
    fig.suptitle("Score Heatmap by Task Type × Metric", fontsize=13, fontweight="bold")

    path = out_dir / "fig5_heatmap_by_type.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Statistical tests (Wilcoxon signed-rank, paired)
# ---------------------------------------------------------------------------

def wilcoxon_tests(df: pd.DataFrame) -> pd.DataFrame:
    pe = df[df["prompt_type"] == "plain_english"].set_index("task_id")
    ps = df[df["prompt_type"] == "promptscript"].set_index("task_id")
    common = pe.index.intersection(ps.index)

    rows = []
    for metric in METRICS:
        a = pe.loc[common, metric].values
        b = ps.loc[common, metric].values
        diff = b - a  # positive = PromptScript better
        if len(diff) < 2 or np.all(diff == 0):
            rows.append({
                "metric": METRIC_LABELS[metric],
                "plain_english_mean": float(np.mean(a)),
                "promptscript_mean": float(np.mean(b)),
                "mean_diff (PS-PE)": float(np.mean(diff)),
                "p_value": float("nan"),
                "significant (p<0.05)": False,
            })
            continue
        stat, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        rows.append({
            "metric": METRIC_LABELS[metric],
            "plain_english_mean": float(np.mean(a)),
            "promptscript_mean": float(np.mean(b)),
            "mean_diff (PS-PE)": float(np.mean(diff)),
            "p_value": round(float(p), 4),
            "significant (p<0.05)": bool(p < 0.05),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task_type in TASK_TYPE_ORDER:
        sub = df[df["task_type"] == task_type]
        if sub.empty:
            continue
        for pt in ["plain_english", "promptscript"]:
            s = sub[sub["prompt_type"] == pt]
            if s.empty:
                continue
            row = {"task_type": task_type, "prompt_type": pt, "n": len(s)}
            for m in METRICS:
                row[f"{m}_mean"] = round(s[m].mean(), 4)
                row[f"{m}_std"] = round(s[m].std(), 4)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--results", default="benchmark/results/results.jsonl", show_default=True)
@click.option("--output", default="report/figures", show_default=True)
def main(results: str, output: str) -> None:
    """Generate all charts and statistical tests from benchmark results."""
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results} ...")
    df = load_results(results)
    print(f"  {len(df)} rows, {df['task_id'].nunique()} tasks, "
          f"prompt types: {df['prompt_type'].unique().tolist()}")

    print("\nGenerating figures ...")
    fig_overall_bars(df, out_dir)
    fig_correctness_by_type(df, out_dir)
    fig_scatter_correctness_faithfulness(df, out_dir)
    fig_token_histogram(df, out_dir)
    fig_heatmap_by_type(df, out_dir)

    print("\nSummary table (mean ± std):")
    tbl = summary_table(df)
    print(tbl.to_string(index=False))
    tbl.to_csv(out_dir / "summary_table.csv", index=False)
    print(f"  Saved {out_dir}/summary_table.csv")

    print("\nWilcoxon signed-rank tests (PromptScript vs Plain English):")
    wtbl = wilcoxon_tests(df)
    print(wtbl.to_string(index=False))
    wtbl.to_csv(out_dir / "wilcoxon_tests.csv", index=False)
    print(f"  Saved {out_dir}/wilcoxon_tests.csv")

    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
