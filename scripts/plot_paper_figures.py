#!/usr/bin/env python
"""D4 — Plot generation for paper figures.

Reads the comparison table from scorecards and produces paper-ready charts.

Usage:
    python scripts/plot_paper_figures.py
    python scripts/plot_paper_figures.py --scorecards_dir outputs/scorecards --out_dir outputs/figures
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Paper-friendly style
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

CATEGORY_ORDER = [
    "Brazil / exams / culture",
    "Social / safety",
    "Text understanding / QA / classification",
    "Reasoning",
    "Math",
    "Code / other",
]

CATEGORY_SHORT = {
    "Brazil / exams / culture": "Brazil/Exams",
    "Social / safety": "Social/Safety",
    "Text understanding / QA / classification": "TextUnd./QA",
    "Reasoning": "Reasoning",
    "Math": "Math",
    "Code / other": "Code/Other",
}

COLORS_CHECKPOINTS = ["#4878d0", "#ee854a", "#6acc64", "#d65f5f", "#956cb4"]
COLORS_BARS = {"all_score": "#4878d0", "native_score": "#ee854a", "translated_score": "#6acc64"}
CHECKPOINT_TICK_LABELS = {
    "Qwen 1.7B Base": "Qwen 1.7B Base",
    "Gigaverbo adapted": "Gigaverbo adapted",
    "Carolina adapted": "Carolina adapted",
}


def _tick_label(checkpoint: str) -> str:
    return CHECKPOINT_TICK_LABELS.get(checkpoint, checkpoint)


def _slugify(value: str) -> str:
    return value.replace(" ", "_").replace("/", "_")


def plot_all_native_translated(comp: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart: checkpoint vs All / Native / Translated scores."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = range(len(comp))
    width = 0.25

    for i, (col, label) in enumerate([
        ("all_score", "All"),
        ("native_score", "Native"),
        ("translated_score", "Translated"),
    ]):
        vals = comp[col].tolist()
        bars = ax.bar(
            [xi + i * width for xi in x],
            vals,
            width,
            label=label,
            color=COLORS_BARS[col],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            if val is not None and not pd.isna(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels([_tick_label(cp) for cp in comp["checkpoint"]], rotation=0, ha="center")
    ax.set_ylabel("Score (accuracy)")
    ax.set_title("PoETa v2: All vs Native vs Translated")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8, borderaxespad=0.2)
    ax.set_ylim(0, min(1.0, comp[["all_score", "native_score", "translated_score"]].max().max() + 0.10))
    ax.grid(axis="y", alpha=0.3)

    out_path = out_dir / "all_native_translated.png"
    fig.savefig(out_path)
    plt.close(fig)
    log.info("Wrote: %s", out_path)

    # Also save PDF
    fig2, ax2 = plt.subplots(figsize=(7, 3.2))
    for i, (col, label) in enumerate([
        ("all_score", "All"),
        ("native_score", "Native"),
        ("translated_score", "Translated"),
    ]):
        vals = comp[col].tolist()
        ax2.bar([xi + i * width for xi in x], vals, width, label=label, color=COLORS_BARS[col], edgecolor="white", linewidth=0.5)
    ax2.set_xticks([xi + width for xi in x])
    ax2.set_xticklabels([_tick_label(cp) for cp in comp["checkpoint"]], rotation=0, ha="center")
    ax2.set_ylabel("Score (accuracy)")
    ax2.set_title("PoETa v2: All vs Native vs Translated")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8, borderaxespad=0.2)
    ax2.set_ylim(0, min(1.0, comp[["all_score", "native_score", "translated_score"]].max().max() + 0.10))
    ax2.grid(axis="y", alpha=0.3)
    fig2.savefig(out_dir / "all_native_translated.pdf")
    plt.close(fig2)


def plot_category_breakdown(comp: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart: checkpoint vs category scores."""
    cats = [c for c in CATEGORY_ORDER if c in comp.columns]
    if not cats:
        log.warning("No category columns found in comparison table")
        return

    fig, ax = plt.subplots(figsize=(9, 3.6))
    n_checkpoints = len(comp)
    n_cats = len(cats)
    width = 0.8 / n_checkpoints
    x = range(n_cats)

    for j, (_, row) in enumerate(comp.iterrows()):
        vals = [row.get(c, 0) for c in cats]
        positions = [xi + j * width - (n_checkpoints - 1) * width / 2 for xi in x]
        color = COLORS_CHECKPOINTS[j % len(COLORS_CHECKPOINTS)]
        bars = ax.bar(positions, vals, width, label=_tick_label(row["checkpoint"]), color=color, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val and not pd.isna(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=45,
                )

    ax.set_xticks(list(x))
    ax.set_xticklabels([CATEGORY_SHORT.get(c, c) for c in cats], rotation=0, ha="center")
    ax.set_ylabel("Score (accuracy)")
    ax.set_title("PoETa v2: Category Breakdown by Checkpoint")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8, borderaxespad=0.2)
    ax.set_ylim(0, max(comp[cats].max()) + 0.08)
    ax.grid(axis="y", alpha=0.3)

    out_path = out_dir / "category_breakdown.png"
    fig.savefig(out_path)
    plt.close(fig)
    log.info("Wrote: %s", out_path)

    # PDF
    fig.savefig(out_dir / "category_breakdown.pdf") if False else None  # already closed
    # Re-create for PDF
    fig2, ax2 = plt.subplots(figsize=(9, 3.6))
    for j, (_, row) in enumerate(comp.iterrows()):
        vals = [row.get(c, 0) for c in cats]
        positions = [xi + j * width - (n_checkpoints - 1) * width / 2 for xi in x]
        ax2.bar(positions, vals, width, label=_tick_label(row["checkpoint"]), color=COLORS_CHECKPOINTS[j % len(COLORS_CHECKPOINTS)], edgecolor="white", linewidth=0.5)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([CATEGORY_SHORT.get(c, c) for c in cats], rotation=0, ha="center")
    ax2.set_ylabel("Score (accuracy)")
    ax2.set_title("PoETa v2: Category Breakdown by Checkpoint")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8, borderaxespad=0.2)
    ax2.set_ylim(0, max(comp[cats].max()) + 0.08)
    ax2.grid(axis="y", alpha=0.3)
    fig2.savefig(out_dir / "category_breakdown.pdf")
    plt.close(fig2)


def plot_task_deltas(scorecards_dir: Path, out_dir: Path) -> None:
    """Horizontal bar chart showing top task deltas between checkpoints."""
    delta_path = scorecards_dir / "task_deltas.csv"
    if not delta_path.exists():
        log.info("No task_deltas.csv found, skipping delta plot")
        return

    df = pd.read_csv(delta_path)
    pairs = df[["from_checkpoint", "to_checkpoint"]].drop_duplicates().values

    for cp_from, cp_to in pairs:
        subset = df[(df["from_checkpoint"] == cp_from) & (df["to_checkpoint"] == cp_to)].copy()
        subset = subset.sort_values("delta")

        # Show top 5 + bottom 5
        show = pd.concat([subset.head(5), subset.tail(5)]).drop_duplicates()
        show = show.sort_values("delta")

        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ["#d65f5f" if d < 0 else "#6acc64" for d in show["delta"]]
        ax.barh(range(len(show)), show["delta"], color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(show)))
        ax.set_yticklabels(show["task"], fontsize=8)
        ax.set_xlabel("Score delta")
        ax.set_title(f"Task deltas: {cp_from} -> {cp_to}")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.grid(axis="x", alpha=0.3)

        fname = f"task_deltas_{_slugify(cp_from)}_to_{_slugify(cp_to)}.png"
        fig.savefig(out_dir / fname)
        plt.close(fig)
        log.info("Wrote: %s", out_dir / fname)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate paper figures from scorecards")
    p.add_argument("--scorecards_dir", default="outputs/scorecards")
    p.add_argument("--out_dir", default="outputs/figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scorecards_dir = Path(args.scorecards_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comp_path = scorecards_dir / "comparison_table.csv"
    if not comp_path.exists():
        log.error("comparison_table.csv not found in %s", scorecards_dir)
        log.error("Run generate_scorecards.py first.")
        return

    comp = pd.read_csv(comp_path)
    log.info("Loaded comparison table: %d checkpoints", len(comp))

    plot_all_native_translated(comp, out_dir)
    plot_category_breakdown(comp, out_dir)
    plot_task_deltas(scorecards_dir, out_dir)

    log.info("All figures saved to %s", out_dir)


if __name__ == "__main__":
    main()
