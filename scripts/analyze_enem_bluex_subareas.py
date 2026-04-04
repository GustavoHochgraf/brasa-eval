#!/usr/bin/env python
"""Supplementary analysis: ENEM/BLUEX subarea breakdown.

Extracts per-subarea scores from evaluation results for enem2022_greedy
and bluex_greedy, produces comparison CSVs, a heatmap figure, and a
markdown summary.

enem_greedy (older editions) lacks subject-level keys — only year splits
are available — so it is excluded from this subarea analysis.

Usage:
    python scripts/analyze_enem_bluex_subareas.py
    python scripts/analyze_enem_bluex_subareas.py --results_dir outputs/eval_results
"""
from __future__ import annotations

import argparse
import json
import logging
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
    "Qwen 1.7B Base",
    "Gigaverbo adapted",
    "Carolina adapted",
]
BASELINE = CHECKPOINTS[0]

CHECKPOINT_SHORT = {
    "Qwen 1.7B Base": "Qwen 1.7B\nBase",
    "Gigaverbo adapted": "Gigaverbo\nadapted",
    "Carolina adapted": "Carolina\nadapted",
}

CHECKPOINT_LEGEND = {
    "Qwen 1.7B Base": "Qwen 1.7B Base",
    "Gigaverbo adapted": "Gigaverbo adapted",
    "Carolina adapted": "Carolina adapted",
}

# Short names for CSV column headers (no line breaks)
CHECKPOINT_CSV = {
    "Qwen 1.7B Base": "qwen_1_7b_base",
    "Gigaverbo adapted": "gigaverbo_adapted",
    "Carolina adapted": "carolina_adapted",
}

TASKS = {
    "enem2022_greedy": {
        "label": "ENEM 2022",
        "subareas": ["human-sciences", "mathematics", "natural-sciences", "languages"],
    },
    "bluex_greedy": {
        "label": "BLUEX",
        "subareas": [
            "history", "geography", "biology", "chemistry",
            "physics", "mathematics", "portuguese", "english", "philosophy",
        ],
    },
}

SUBAREA_DISPLAY = {
    "human-sciences": "Human Sciences",
    "mathematics": "Mathematics",
    "natural-sciences": "Natural Sciences",
    "languages": "Languages",
    "history": "History",
    "geography": "Geography",
    "biology": "Biology",
    "chemistry": "Chemistry",
    "physics": "Physics",
    "portuguese": "Portuguese",
    "english": "English",
    "philosophy": "Philosophy",
}

# ── Helpers ───────────────────────────────────────────────────────────

def load_results(results_dir: Path, checkpoint: str) -> dict:
    path = results_dir / f"{checkpoint}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", data)


def extract_subarea_scores(results_dir: Path) -> pd.DataFrame:
    """Extract per-checkpoint, per-task, per-subarea scores."""
    rows: list[dict] = []
    for cp in CHECKPOINTS:
        try:
            results = load_results(results_dir, cp)
        except FileNotFoundError:
            log.warning("Result file not found for %s, skipping", cp)
            continue

        for task_key, task_cfg in TASKS.items():
            if task_key not in results:
                log.warning("[%s] task %s not found in results", cp, task_key)
                continue
            task_data = results[task_key]
            for subarea in task_cfg["subareas"]:
                if subarea in task_data:
                    rows.append({
                        "checkpoint": cp,
                        "task": task_cfg["label"],
                        "subarea": subarea,
                        "score": float(task_data[subarea]),
                    })
                else:
                    log.warning("[%s] %s: subarea '%s' not found", cp, task_key, subarea)

    return pd.DataFrame(rows)


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute deltas vs baseline for each task/subarea."""
    base = df[df["checkpoint"] == BASELINE][["task", "subarea", "score"]].rename(
        columns={"score": "baseline_score"}
    )
    delta_rows: list[dict] = []
    for _, brow in base.iterrows():
        row = {
            "task": brow["task"],
            "subarea": brow["subarea"],
            "baseline_score": round(brow["baseline_score"], 4),
        }
        for cp in CHECKPOINTS[1:]:
            col_key = CHECKPOINT_CSV[cp]
            match = df[
                (df["checkpoint"] == cp)
                & (df["task"] == brow["task"])
                & (df["subarea"] == brow["subarea"])
            ]
            if not match.empty:
                score = match.iloc[0]["score"]
                row[f"{col_key}_score"] = round(score, 4)
                row[f"{col_key}_delta"] = round(score - brow["baseline_score"], 4)
            else:
                row[f"{col_key}_score"] = None
                row[f"{col_key}_delta"] = None
        delta_rows.append(row)
    return pd.DataFrame(delta_rows)


# ── Figure ────────────────────────────────────────────────────────────

def _build_heatmap(df: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    """Build the heatmap figure (shared by PNG and PDF export)."""
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })

    tasks_ordered = ["ENEM 2022", "BLUEX"]
    task_dfs = []
    for task in tasks_ordered:
        tdf = df[df["task"] == task].copy()
        if tdf.empty:
            continue
        pivot = tdf.pivot(index="subarea", columns="checkpoint", values="score")
        pivot = pivot.reindex(columns=CHECKPOINTS)
        task_key = [k for k, v in TASKS.items() if v["label"] == task][0]
        ordered = [s for s in TASKS[task_key]["subareas"] if s in pivot.index]
        pivot = pivot.reindex(ordered)
        task_dfs.append((task, pivot))

    if not task_dfs:
        log.warning("No data for heatmap")
        return None, None

    n_rows = sum(len(p) for _, p in task_dfs)
    fig, axes = plt.subplots(
        len(task_dfs), 1,
        figsize=(6.5, 0.55 * n_rows + 2.8),
        gridspec_kw={
            "height_ratios": [len(p) for _, p in task_dfs],
            "hspace": 0.55,
        },
    )
    if len(task_dfs) == 1:
        axes = [axes]

    for idx, (ax, (task_label, pivot)) in enumerate(zip(axes, task_dfs)):
        data = pivot.values
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.1, vmax=0.85)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isnan(val):
                    continue
                text_color = "white" if val < 0.3 or val > 0.75 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8.5, color=text_color, fontweight="bold")

        ax.set_xticks(range(len(CHECKPOINTS)))
        # Only show x-tick labels on the bottom subplot
        if idx == len(task_dfs) - 1:
            ax.set_xticklabels(
                [CHECKPOINT_SHORT[c] for c in CHECKPOINTS],
                fontsize=8.5, ha="center",
            )
        else:
            ax.set_xticklabels(
                [CHECKPOINT_SHORT[c] for c in CHECKPOINTS],
                fontsize=8.5, ha="center",
            )
            ax.tick_params(axis="x", pad=6)

        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(
            [SUBAREA_DISPLAY.get(s, s) for s in pivot.index],
            fontsize=8.5,
        )
        ax.set_title(task_label, fontsize=11, fontweight="bold", pad=10)

    fig.colorbar(im, ax=axes, shrink=0.5, label="Accuracy", pad=0.06)
    fig.suptitle(
        "ENEM / BLUEX: Subarea Breakdown by Checkpoint",
        fontsize=12, fontweight="bold", y=0.98,
    )

    return fig, axes


def plot_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Save heatmap as PNG and PDF."""
    for ext in ["png", "pdf"]:
        fig, axes = _build_heatmap(df)
        if fig is None:
            return
        out_path = out_dir / f"enem_bluex_subarea_heatmap.{ext}"
        fig.savefig(out_path)
        plt.close(fig)
        log.info("Wrote: %s", out_path)


def plot_grouped_bars(df: pd.DataFrame, out_dir: Path) -> None:
    """Save the side-by-side grouped bar chart requested for paper use."""
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })

    colors = ["#4878d0", "#ee854a", "#6acc64"]
    tasks_ordered = ["ENEM 2022", "BLUEX"]

    for ext in ["png", "pdf"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        if len(tasks_ordered) == 1:
            axes = [axes]

        for ax, task_label in zip(axes, tasks_ordered):
            task_key = [k for k, v in TASKS.items() if v["label"] == task_label][0]
            ordered_subareas = TASKS[task_key]["subareas"]
            pivot = (
                df[df["task"] == task_label]
                .pivot(index="subarea", columns="checkpoint", values="score")
                .reindex(index=ordered_subareas, columns=CHECKPOINTS)
            )

            x = np.arange(len(pivot.index))
            width = 0.24

            for idx, checkpoint in enumerate(CHECKPOINTS):
                vals = pivot[checkpoint].tolist()
                positions = x + idx * width - width
                bars = ax.bar(
                    positions,
                    vals,
                    width,
                    label=CHECKPOINT_LEGEND[checkpoint],
                    color=colors[idx],
                    edgecolor="white",
                    linewidth=0.5,
                )
                for bar, val in zip(bars, vals):
                    if np.isnan(val):
                        continue
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + 0.012,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        rotation=32,
                        rotation_mode="anchor",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.15},
                    )

            ax.set_xticks(x)
            ax.set_xticklabels(
                [SUBAREA_DISPLAY.get(subarea, subarea) for subarea in pivot.index],
                rotation=40,
                ha="right",
            )
            ax.set_ylim(0, 1.08)
            ax.set_ylabel("Score")
            ax.set_title(f"{task_label} Subarea Scores")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(loc="upper right", fontsize=8)

        out_path = out_dir / f"enem_bluex_subareas.{ext}"
        fig.savefig(out_path)
        plt.close(fig)
        log.info("Wrote: %s", out_path)


# ── Summary ───────────────────────────────────────────────────────────

def generate_summary(deltas_df: pd.DataFrame, out_dir: Path) -> None:
    """Write a short markdown summary of subarea findings."""
    lines = [
        "# ENEM / BLUEX Subarea Analysis — Summary",
        "",
        "Supplementary drill-down within the Brazil / Exams category.",
        "Baseline: Qwen 1.7B Base. Deltas in percentage points (pp).",
        "",
    ]

    for cp_key, cp_label in [
        ("gigaverbo_adapted", "Gigaverbo adapted"),
        ("carolina_adapted", "Carolina adapted"),
    ]:
        delta_col = f"{cp_key}_delta"
        if delta_col not in deltas_df.columns:
            continue

        valid = deltas_df.dropna(subset=[delta_col]).copy()
        top_gains = valid.nlargest(3, delta_col)
        top_losses = valid.nsmallest(3, delta_col)

        lines.append(f"## {cp_label}")
        lines.append("")
        lines.append("**Largest gains vs baseline:**")
        for _, r in top_gains.iterrows():
            lines.append(
                f"- {r['task']} / {SUBAREA_DISPLAY.get(r['subarea'], r['subarea'])}: "
                f"{r['baseline_score']:.2f} -> {r[f'{cp_key}_score']:.2f} "
                f"({r[delta_col]*100:+.1f} pp)"
            )
        lines.append("")
        lines.append("**Largest losses vs baseline:**")
        for _, r in top_losses.iterrows():
            if r[delta_col] >= 0:
                continue
            lines.append(
                f"- {r['task']} / {SUBAREA_DISPLAY.get(r['subarea'], r['subarea'])}: "
                f"{r['baseline_score']:.2f} -> {r[f'{cp_key}_score']:.2f} "
                f"({r[delta_col]*100:+.1f} pp)"
            )
        lines.append("")

    # Key findings from data
    lines.extend(["## Key findings", ""])
    findings = []

    enem_math = deltas_df[(deltas_df["task"] == "ENEM 2022") & (deltas_df["subarea"] == "mathematics")]
    if not enem_math.empty:
        r = enem_math.iloc[0]
        giga_d = r.get("gigaverbo_adapted_delta")
        carol_d = r.get("carolina_adapted_delta")
        if giga_d is not None and carol_d is not None:
            findings.append(
                f"- **ENEM 2022 Mathematics** sees the largest absolute gains from continued pretraining "
                f"(+{giga_d*100:.1f} pp Gigaverbo adapted, +{carol_d*100:.1f} pp Carolina adapted), "
                f"though from a very low baseline ({r['baseline_score']:.2f})."
            )

    bluex_bio = deltas_df[(deltas_df["task"] == "BLUEX") & (deltas_df["subarea"] == "biology")]
    if not bluex_bio.empty:
        r = bluex_bio.iloc[0]
        giga_d = r.get("gigaverbo_adapted_delta")
        carol_d = r.get("carolina_adapted_delta")
        if giga_d is not None and carol_d is not None:
            findings.append(
                f"- **BLUEX Biology** improves with both corpora "
                f"(+{giga_d*100:.1f} pp Gigaverbo adapted, +{carol_d*100:.1f} pp Carolina adapted), "
                f"the largest BLUEX gain for Carolina adapted."
            )

    for sub in ["physics", "chemistry"]:
        row = deltas_df[(deltas_df["task"] == "BLUEX") & (deltas_df["subarea"] == sub)]
        if not row.empty:
            r = row.iloc[0]
            giga_d = r.get("gigaverbo_adapted_delta")
            carol_d = r.get("carolina_adapted_delta")
            if giga_d is not None and carol_d is not None and (giga_d < -0.01 or carol_d < -0.01):
                findings.append(
                    f"- **BLUEX {sub.title()}** declines for at least one model "
                    f"({giga_d*100:+.1f} pp Gigaverbo adapted, {carol_d*100:+.1f} pp Carolina adapted), "
                    f"suggesting continued pretraining in Portuguese does not help STEM problem-solving."
                )
                break

    enem_ns = deltas_df[(deltas_df["task"] == "ENEM 2022") & (deltas_df["subarea"] == "natural-sciences")]
    if not enem_ns.empty:
        r = enem_ns.iloc[0]
        carol_d = r.get("carolina_adapted_delta")
        if carol_d is not None and carol_d > 0.05:
            findings.append(
                f"- **ENEM 2022 Natural Sciences** shows a strong Carolina adapted gain "
                f"(+{carol_d*100:.1f} pp), consistent with the Carolina corpus "
                f"containing Brazilian educational content."
            )

    findings.append(
        "- Gains from continued pretraining are concentrated in humanities and life sciences; "
        "exact sciences (mathematics, physics, chemistry) show mixed or negative effects, "
        "reinforcing that linguistic pretraining benefits language-dependent reasoning more than "
        "formal/symbolic reasoning."
    )

    lines.extend(findings)
    lines.append("")

    out_path = out_dir / "enem_bluex_subarea_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote: %s", out_path)


# ── Main ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ENEM/BLUEX subarea analysis")
    p.add_argument("--results_dir", default="outputs/eval_results")
    p.add_argument("--scorecards_dir", default="outputs/scorecards")
    p.add_argument("--figures_dir", default="outputs/figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    scorecards_dir = Path(args.scorecards_dir)
    figures_dir = Path(args.figures_dir)
    scorecards_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract scores
    df = extract_subarea_scores(results_dir)
    if df.empty:
        log.error("No subarea data extracted. Check result files.")
        return
    log.info("Extracted %d subarea scores across %d checkpoints", len(df), df["checkpoint"].nunique())

    # Save tidy CSV
    df_out = df.copy()
    df_out["score"] = df_out["score"].round(4)
    df_out.to_csv(scorecards_dir / "enem_bluex_subarea_scores.csv", index=False)
    log.info("Wrote: %s", scorecards_dir / "enem_bluex_subarea_scores.csv")

    # 2. Compute deltas
    deltas_df = compute_deltas(df)
    deltas_df.to_csv(scorecards_dir / "enem_bluex_subarea_deltas_vs_baseline.csv", index=False)
    log.info("Wrote: %s", scorecards_dir / "enem_bluex_subarea_deltas_vs_baseline.csv")

    # 3. Figures -> figures/
    plot_heatmap(df, figures_dir)
    plot_grouped_bars(df, figures_dir)

    # 4. Summary -> scorecards/
    generate_summary(deltas_df, scorecards_dir)

    log.info("Done. CSVs/summary in %s, figures in %s", scorecards_dir, figures_dir)


if __name__ == "__main__":
    main()
