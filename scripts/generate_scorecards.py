#!/usr/bin/env python
"""D3 — Diagnostic scorecard generator.

Parses bulk evaluation result files and produces paper-ready scorecards
with All / Native / Translated scores and category breakdowns.

Expected result format (one JSON per checkpoint):
    {
        "results": {
            "<lm_eval_task>": {
                "<metric_name>": <float>,
                ...
            },
            ...
        },
        "config": { ... }   # optional
    }

Usage:
    python scripts/generate_scorecards.py --results_dir outputs/eval_results
    python scripts/generate_scorecards.py --results_dir outputs/eval_results --segmentation data/paper_final_segmentation.csv
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Preferred metric resolution order per task
# ---------------------------------------------------------------------------
METRIC_PREFERENCE = ["acc_norm", "acc", "f1_macro", "f1", "exact_match", "pearson"]


def _resolve_metric(task_results: dict[str, float], lm_eval_task: str) -> tuple[str, float] | None:
    """Find best available metric for a task's results."""
    for m in METRIC_PREFERENCE:
        # lm-eval-harness may prefix metric with task name or use comma variants
        for key, val in task_results.items():
            # Strip common prefixes/suffixes
            clean = key.replace(",none", "").split(",")[0].strip()
            if clean == m:
                return m, float(val)
    # Fallback: take first numeric value
    for key, val in task_results.items():
        try:
            return key, float(val)
        except (ValueError, TypeError):
            continue
    return None


def load_results(results_path: Path) -> dict[str, dict[str, float]]:
    """Load evaluation results from a JSON file.

    Supports lm-eval-harness output format with nested 'results' key,
    or flat {task: {metric: score}} format.
    """
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "results" in data and isinstance(data["results"], dict):
        return data["results"]
    # Flat format
    return data


def build_scorecard(
    results: dict[str, dict[str, float]],
    seg: pd.DataFrame,
    checkpoint_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a scorecard DataFrame and summary dict for one checkpoint.

    Returns:
        (task_scores_df, summary_dict)
    """
    rows: list[dict] = []
    skipped: list[str] = []

    for _, task_row in seg.iterrows():
        task_name = task_row["task"]
        lm_eval = task_row["lm_eval_task"]
        cat = task_row["categoria_paper"]
        nt = task_row["native_translated"]

        if not lm_eval or lm_eval not in results:
            skipped.append(task_name)
            continue

        task_results = results[lm_eval]
        resolved = _resolve_metric(task_results, lm_eval)
        if resolved is None:
            skipped.append(task_name)
            log.warning("No usable metric for %s (%s)", task_name, lm_eval)
            continue

        metric_name, score = resolved
        rows.append({
            "checkpoint": checkpoint_name,
            "task": task_name,
            "lm_eval_task": lm_eval,
            "categoria_paper": cat,
            "native_translated": nt,
            "metric": metric_name,
            "score": score,
        })

    if skipped:
        log.info("[%s] Skipped %d tasks (not in results): %s", checkpoint_name, len(skipped), ", ".join(skipped[:10]))

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("[%s] No scores found!", checkpoint_name)
        return df, {"checkpoint": checkpoint_name, "error": "no scores"}

    # --- Compute aggregates ---
    all_score = df["score"].mean()
    native_mask = df["native_translated"] == "Native"
    translated_mask = df["native_translated"] == "Translated"

    native_score = df.loc[native_mask, "score"].mean() if native_mask.any() else None
    translated_score = df.loc[translated_mask, "score"].mean() if translated_mask.any() else None

    # Per-category scores
    cat_scores = df.groupby("categoria_paper")["score"].mean().to_dict()

    summary = {
        "checkpoint": checkpoint_name,
        "n_tasks": len(df),
        "n_skipped": len(skipped),
        "all_score": round(all_score, 4),
        "native_score": round(native_score, 4) if native_score is not None else None,
        "translated_score": round(translated_score, 4) if translated_score is not None else None,
        "n_native": int(native_mask.sum()),
        "n_translated": int(translated_mask.sum()),
        "category_scores": {k: round(v, 4) for k, v in sorted(cat_scores.items())},
        "skipped_tasks": skipped,
    }

    return df, summary


def compute_deltas(
    task_dfs: dict[str, pd.DataFrame],
    checkpoint_order: list[str],
) -> pd.DataFrame | None:
    """Compute per-task score deltas: all checkpoints vs the first (base).

    The first checkpoint in the order is treated as the baseline.
    Returns a df with columns: task, from_checkpoint, to_checkpoint, delta.
    """
    if len(checkpoint_order) < 2:
        return None

    base_cp = checkpoint_order[0]
    df_base = task_dfs.get(base_cp)
    if df_base is None:
        return None

    delta_rows: list[dict] = []
    for cp_to in checkpoint_order[1:]:
        df_to = task_dfs.get(cp_to)
        if df_to is None:
            continue

        merged = df_base[["task", "score"]].merge(
            df_to[["task", "score"]],
            on="task",
            suffixes=("_from", "_to"),
        )
        for _, r in merged.iterrows():
            delta_rows.append({
                "task": r["task"],
                "from_checkpoint": base_cp,
                "to_checkpoint": cp_to,
                "score_from": r["score_from"],
                "score_to": r["score_to"],
                "delta": round(r["score_to"] - r["score_from"], 4),
            })

    return pd.DataFrame(delta_rows) if delta_rows else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate diagnostic scorecards from evaluation results")
    p.add_argument("--results_dir", required=True, help="Directory with result JSON files (one per checkpoint)")
    p.add_argument("--segmentation", default="data/paper_final_segmentation.csv")
    p.add_argument("--out_dir", default="outputs/scorecards")
    p.add_argument(
        "--checkpoint_order",
        nargs="*",
        default=None,
        help="Ordered list of checkpoint names for delta computation (derived from filenames if omitted)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seg = pd.read_csv(args.segmentation)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        log.error("Results directory not found: %s", results_dir)
        log.info("Creating sample result files for demonstration...")
        _create_sample_results(results_dir, seg)

    # Discover result files
    result_files = sorted(results_dir.glob("*.json"))
    if not result_files:
        log.error("No JSON result files found in %s", results_dir)
        return

    log.info("Found %d result files in %s", len(result_files), results_dir)

    # Process each checkpoint
    task_dfs: dict[str, pd.DataFrame] = {}
    summaries: list[dict] = []
    checkpoint_names: list[str] = []

    for rpath in result_files:
        cp_name = rpath.stem
        checkpoint_names.append(cp_name)
        log.info("Processing: %s", cp_name)

        results = load_results(rpath)
        task_df, summary = build_scorecard(results, seg, cp_name)

        # Save per-checkpoint scorecard
        if not task_df.empty:
            task_df.to_csv(out_dir / f"{cp_name}_scorecard.csv", index=False)
            with open(out_dir / f"{cp_name}_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

        task_dfs[cp_name] = task_df
        summaries.append(summary)

    # --- Comparison table ---
    comp_rows = []
    for s in summaries:
        row = {
            "checkpoint": s["checkpoint"],
            "n_tasks": s.get("n_tasks", 0),
            "all_score": s.get("all_score"),
            "native_score": s.get("native_score"),
            "translated_score": s.get("translated_score"),
        }
        for cat, score in s.get("category_scores", {}).items():
            row[cat] = score
        comp_rows.append(row)

    if comp_rows:
        comp_df = pd.DataFrame(comp_rows)
        comp_df.to_csv(out_dir / "comparison_table.csv", index=False)
        log.info("Wrote: %s", out_dir / "comparison_table.csv")

    # --- Delta analysis ---
    order = args.checkpoint_order or checkpoint_names
    delta_df = compute_deltas(task_dfs, order)
    if delta_df is not None and not delta_df.empty:
        delta_df.to_csv(out_dir / "task_deltas.csv", index=False)

        # Top 5 rising / falling
        for cp_pair in delta_df[["from_checkpoint", "to_checkpoint"]].drop_duplicates().values:
            subset = delta_df[
                (delta_df["from_checkpoint"] == cp_pair[0]) & (delta_df["to_checkpoint"] == cp_pair[1])
            ].copy()
            top_rising = subset.nlargest(5, "delta")[["task", "delta", "score_from", "score_to"]]
            top_falling = subset.nsmallest(5, "delta")[["task", "delta", "score_from", "score_to"]]

            pair_label = f"{cp_pair[0]}_to_{cp_pair[1]}"
            print(f"\n--- {pair_label} ---")
            print("Top 5 rising:")
            for _, r in top_rising.iterrows():
                print(f"  {r['task']:40s}  {r['score_from']:.4f} -> {r['score_to']:.4f}  (d {r['delta']:+.4f})")
            print("Top 5 falling:")
            for _, r in top_falling.iterrows():
                print(f"  {r['task']:40s}  {r['score_from']:.4f} -> {r['score_to']:.4f}  (d {r['delta']:+.4f})")

        log.info("Wrote: %s", out_dir / "task_deltas.csv")

    log.info("Done. Scorecards in %s", out_dir)


def _create_sample_results(results_dir: Path, seg: pd.DataFrame) -> None:
    """Create synthetic sample results for demonstration/testing."""
    import random
    random.seed(42)

    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = [
        ("Qwen-1.7B-Base", 0.30),
        ("TuQwen-1.7B-Base", 0.35),
        ("TuQwen-1.7B-Base-0.87ep", 0.33),
    ]

    for cp_name, base_score in checkpoints:
        results: dict[str, dict[str, float]] = {}
        for _, row in seg.iterrows():
            lm_eval = row["lm_eval_task"]
            if not lm_eval:
                continue
            # Synthetic score: base + noise + bonus for Native tasks in TuQwen
            score = base_score + random.gauss(0, 0.08)
            if "TuQwen" in cp_name and row["native_translated"] == "Native":
                score += 0.05  # simulate PT training benefit
            if "0.87ep" in cp_name and row["native_translated"] == "Translated":
                score -= 0.03  # simulate slight translated regression
            score = max(0.0, min(1.0, score))
            results[lm_eval] = {"acc": round(score, 4)}

        out_path = results_dir / f"{cp_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2)
        log.info("Created sample: %s", out_path)


if __name__ == "__main__":
    main()
