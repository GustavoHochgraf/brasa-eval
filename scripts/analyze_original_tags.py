#!/usr/bin/env python
"""Complementary analysis using the ORIGINAL PoETa subcategory tags.

The PoETa v2 benchmark assigns multi-label subcategory tags to each task
(e.g., "reasoning, math" or "common-sense, text-understanding").  The paper
collapses these into 6 single-label diagnostic categories for interpretability.

This script preserves the multi-label structure and produces:
  - data/original_poeta_tag_expanded.csv   (one row per task x tag)
  - data/original_poeta_tag_summary.csv    (counts per tag)
  - data/manual_vs_original_taxonomy_map.csv (mapping both taxonomies)
  - outputs/scorecards/original_tag_breakdown.csv (tidy scores per tag)
  - outputs/scorecards/original_tag_breakdown_pivot.csv (pivoted comparison)
  - outputs/figures/original_tag_breakdown_heatmap.{png,pdf}
  - outputs/figures/original_tag_breakdown_by_checkpoint.{png,pdf}
  - outputs/scorecards/original_tag_analysis_note.md

Methodological note:
  Because original tags are multi-label, a single task contributes to every
  tag it belongs to.  This means tag-level means are NOT mutually exclusive
  and should not be summed or averaged across tags.

Usage:
    python scripts/analyze_original_tags.py
    python scripts/analyze_original_tags.py --segmentation data/paper_final_segmentation.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────

CHECKPOINTS = [
    "Qwen3-1.7B-Base",
    "TuQwen3-Base-LR1e5-run1",
    "QwenRolina3-Base",
]

CHECKPOINT_SHORT = {
    "Qwen3-1.7B-Base": "Qwen Base",
    "TuQwen3-Base-LR1e5-run1": "TuQwen\n(GigaVerbo)",
    "QwenRolina3-Base": "QwenRolina\n(Carolina)",
}

CHECKPOINT_SHORT_FLAT = {
    "Qwen3-1.7B-Base": "Qwen Base",
    "TuQwen3-Base-LR1e5-run1": "TuQwen (GigaVerbo)",
    "QwenRolina3-Base": "QwenRolina (Carolina)",
}

# ── Helpers ───────────────────────────────────────────────────────────

def load_results(results_dir: Path, checkpoint: str) -> dict:
    path = results_dir / f"{checkpoint}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", data)


def resolve_score(task_results: dict, expected_metric: str) -> float | None:
    """Extract the expected metric value from lm-eval-harness result dict."""
    for key, val in task_results.items():
        clean = key.replace(",none", "").split(",")[0].strip()
        if clean == expected_metric:
            return float(val)
    return None


def expand_tags(seg: pd.DataFrame) -> pd.DataFrame:
    """Expand multi-label original_subcategories into one row per (task, tag)."""
    rows: list[dict] = []
    for _, r in seg.iterrows():
        raw = str(r["original_subcategories"]).strip()
        if not raw or raw == "nan":
            log.warning("Task '%s' has no original_subcategories", r["task"])
            continue
        tags = [t.strip() for t in raw.split(",") if t.strip()]
        for tag in tags:
            rows.append({
                "task": r["task"],
                "lm_eval_task": r["lm_eval_task"],
                "original_tag": tag,
                "native_translated": r["native_translated"],
                "metric": r["metric"],
                "categoria_paper": r["categoria_paper"],
                "included_in_paper_subset": True,
            })
    return pd.DataFrame(rows)


# ── Phase 2: Tag expansion and summary ────────────────────────────────

def create_tag_artifacts(seg: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Create expanded and summary CSVs for original tags."""
    expanded = expand_tags(seg)
    assert not expanded.empty, "No tags found after expansion"

    # expanded CSV
    expanded.to_csv(data_dir / "original_poeta_tag_expanded.csv", index=False)
    log.info("Wrote: %s (%d rows)", data_dir / "original_poeta_tag_expanded.csv", len(expanded))

    # summary CSV
    summary_rows = []
    for tag, group in expanded.groupby("original_tag"):
        summary_rows.append({
            "original_tag": tag,
            "n_tasks": group["task"].nunique(),
            "n_native": group[group["native_translated"] == "Native"]["task"].nunique(),
            "n_translated": group[group["native_translated"] == "Translated"]["task"].nunique(),
            "tasks": "; ".join(sorted(group["task"].unique())),
        })
    summary = pd.DataFrame(summary_rows).sort_values("n_tasks", ascending=False)
    summary.to_csv(data_dir / "original_poeta_tag_summary.csv", index=False)
    log.info("Wrote: %s (%d tags)", data_dir / "original_poeta_tag_summary.csv", len(summary))

    return expanded


# ── Phase 3: Score breakdown by original tag ──────────────────────────

def create_score_breakdown(
    expanded: pd.DataFrame,
    results_dir: Path,
    out_dir: Path,
) -> pd.DataFrame:
    """Compute mean score per (checkpoint, original_tag)."""
    tidy_rows: list[dict] = []

    for cp in CHECKPOINTS:
        results = load_results(results_dir, cp)
        for tag, group in expanded.groupby("original_tag"):
            scores = []
            n_native = 0
            n_translated = 0
            for _, r in group.drop_duplicates("task").iterrows():
                lm_eval = r["lm_eval_task"]
                if lm_eval not in results:
                    log.warning("[%s] Task '%s' not in results", cp, r["task"])
                    continue
                score = resolve_score(results[lm_eval], r["metric"])
                if score is None:
                    log.warning("[%s] Metric '%s' not found for '%s'", cp, r["metric"], r["task"])
                    continue
                scores.append(score)
                if r["native_translated"] == "Native":
                    n_native += 1
                else:
                    n_translated += 1
            if scores:
                tidy_rows.append({
                    "checkpoint": cp,
                    "original_tag": tag,
                    "mean_score": round(np.mean(scores), 4),
                    "n_tasks": len(scores),
                    "n_native_tasks": n_native,
                    "n_translated_tasks": n_translated,
                })

    tidy = pd.DataFrame(tidy_rows)
    tidy.to_csv(out_dir / "original_tag_breakdown.csv", index=False)
    log.info("Wrote: %s", out_dir / "original_tag_breakdown.csv")

    # Pivot table
    pivot = tidy.pivot(index="original_tag", columns="checkpoint", values="mean_score")
    pivot = pivot.reindex(columns=CHECKPOINTS)
    # Sort by baseline score descending
    pivot = pivot.sort_values(CHECKPOINTS[0], ascending=False)
    pivot.to_csv(out_dir / "original_tag_breakdown_pivot.csv")
    log.info("Wrote: %s", out_dir / "original_tag_breakdown_pivot.csv")

    return tidy


# ── Phase 4: Figures ──────────────────────────────────────────────────

def plot_heatmap(tidy: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap of mean_score by (original_tag, checkpoint)."""
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })

    pivot = tidy.pivot(index="original_tag", columns="checkpoint", values="mean_score")
    pivot = pivot.reindex(columns=CHECKPOINTS)
    pivot = pivot.sort_values(CHECKPOINTS[0], ascending=True)

    for ext in ["png", "pdf"]:
        fig, ax = plt.subplots(figsize=(6.5, 0.55 * len(pivot) + 1.8))
        data = pivot.values

        im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.1, vmax=0.85)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isnan(val):
                    continue
                text_color = "white" if val < 0.3 or val > 0.75 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")

        ax.set_xticks(range(len(CHECKPOINTS)))
        ax.set_xticklabels(
            [CHECKPOINT_SHORT[c] for c in CHECKPOINTS],
            fontsize=9, ha="center",
        )
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_title("PoETa v2: Original Tag Breakdown by Checkpoint",
                      fontsize=12, fontweight="bold", pad=12)
        fig.colorbar(im, ax=ax, shrink=0.7, label="Score", pad=0.06)

        out_path = out_dir / f"original_tag_breakdown_heatmap.{ext}"
        fig.savefig(out_path)
        plt.close(fig)
        log.info("Wrote: %s", out_path)


def plot_grouped_bars(tidy: pd.DataFrame, out_dir: Path) -> None:
    """Horizontal grouped bar chart of tag scores across checkpoints."""
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })

    pivot = tidy.pivot(index="original_tag", columns="checkpoint", values="mean_score")
    pivot = pivot.reindex(columns=CHECKPOINTS)
    pivot = pivot.sort_values(CHECKPOINTS[0], ascending=True)

    colors = ["#4878d0", "#ee854a", "#6acc64"]
    n_tags = len(pivot)
    n_cps = len(CHECKPOINTS)
    bar_height = 0.8 / n_cps

    for ext in ["png", "pdf"]:
        fig, ax = plt.subplots(figsize=(8, 0.45 * n_tags + 2))
        y = np.arange(n_tags)

        for j, cp in enumerate(CHECKPOINTS):
            vals = pivot[cp].values
            positions = y + j * bar_height - (n_cps - 1) * bar_height / 2
            ax.barh(positions, vals, bar_height, label=CHECKPOINT_SHORT_FLAT[cp],
                    color=colors[j], edgecolor="white", linewidth=0.5)

        ax.set_yticks(y)
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_xlabel("Score")
        ax.set_title("PoETa v2: Score by Original Tag",
                      fontsize=12, fontweight="bold", pad=10)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim(0, min(1.0, pivot.max().max() + 0.1))
        ax.grid(axis="x", alpha=0.3)

        out_path = out_dir / f"original_tag_breakdown_by_checkpoint.{ext}"
        fig.savefig(out_path)
        plt.close(fig)
        log.info("Wrote: %s", out_path)


# ── Phase 5: Taxonomy mapping and methodological note ─────────────────

def create_taxonomy_map(expanded: pd.DataFrame, data_dir: Path) -> None:
    """Map each task to both its original tags and manual paper category."""
    rows = []
    for task, group in expanded.groupby("task"):
        rows.append({
            "task": task,
            "original_poeta_tags": ", ".join(sorted(group["original_tag"].unique())),
            "manual_paper_category": group.iloc[0]["categoria_paper"],
            "native_translated": group.iloc[0]["native_translated"],
        })
    df = pd.DataFrame(rows).sort_values(["manual_paper_category", "task"])
    df.to_csv(data_dir / "manual_vs_original_taxonomy_map.csv", index=False)
    log.info("Wrote: %s", data_dir / "manual_vs_original_taxonomy_map.csv")


def create_analysis_note(out_dir: Path) -> None:
    """Write a short methodological note."""
    note = """# Original PoETa Tag Analysis — Methodological Note

## Purpose

This analysis complements the paper's manual diagnostic grouping (6 categories)
with a breakdown using the **original PoETa v2 subcategory tags**.

## Key differences

| Aspect | Manual paper grouping | Original PoETa tags |
|--------|----------------------|---------------------|
| Cardinality | 6 categories | 11 tags |
| Label type | Single-label per task | Multi-label per task |
| Design goal | Interpretive diagnostic buckets for the paper | Benchmark-native capability taxonomy |
| Granularity | Coarser, oriented to training/evaluation decisions | Finer, oriented to capability coverage |

## Multi-label handling

Because original PoETa tags are multi-label (e.g., BB Mathematical Induction
has both "reasoning" and "math"), each task contributes to the mean score of
**every tag it belongs to**.  This means:

- Tag-level means are **not mutually exclusive**
- They should **not be summed** or averaged across tags
- The total "task count" across all tags exceeds 40

This is by design: it preserves the original benchmark's capability taxonomy
without forcing an artificial single-label reduction.

## Relationship to the paper

The paper uses the manual grouping for its main results tables and discussion.
The original-tag analysis is presented as a complementary view that:

1. Shows the benchmark's native structure
2. Demonstrates that even finer-grained breakdowns reveal training effects
3. Validates that the manual grouping captures the dominant patterns

## Files

- `data/original_poeta_tag_expanded.csv` — one row per (task, tag) pair
- `data/original_poeta_tag_summary.csv` — tag counts and task lists
- `data/manual_vs_original_taxonomy_map.csv` — both taxonomies side by side
- `outputs/scorecards/original_tag_breakdown.csv` — tidy scores per (checkpoint, tag)
- `outputs/scorecards/original_tag_breakdown_pivot.csv` — pivoted comparison
- `outputs/figures/original_tag_breakdown_heatmap.{png,pdf}` — heatmap
- `outputs/figures/original_tag_breakdown_by_checkpoint.{png,pdf}` — bar chart
"""
    out_path = out_dir / "original_tag_analysis_note.md"
    out_path.write_text(note, encoding="utf-8")
    log.info("Wrote: %s", out_path)


# ── Main ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Original PoETa tag analysis")
    p.add_argument("--segmentation", default="data/paper_final_segmentation.csv")
    p.add_argument("--results_dir", default="outputs/eval_results")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--scorecards_dir", default="outputs/scorecards")
    p.add_argument("--figures_dir", default="outputs/figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seg = pd.read_csv(args.segmentation)
    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)
    scorecards_dir = Path(args.scorecards_dir)
    figures_dir = Path(args.figures_dir)

    for d in [data_dir, scorecards_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        log.error("Results directory not found: %s", results_dir)
        sys.exit(1)

    # Phase 2: Tag expansion
    expanded = create_tag_artifacts(seg, data_dir)

    # Phase 3: Score breakdown
    tidy = create_score_breakdown(expanded, results_dir, scorecards_dir)

    # Phase 4: Figures
    plot_heatmap(tidy, figures_dir)
    plot_grouped_bars(tidy, figures_dir)

    # Phase 5: Taxonomy map and note
    create_taxonomy_map(expanded, data_dir)
    create_analysis_note(scorecards_dir)

    log.info("Done. Original tag analysis complete.")


if __name__ == "__main__":
    main()
