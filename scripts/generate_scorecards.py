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
import sys
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_ORDER = [
    "Qwen 1.7B Base",
    "Gigaverbo adapted",
    "Carolina adapted",
]


def _resolve_metric(
    task_results: dict[str, float],
    lm_eval_task: str,
    expected_metric: str,
) -> tuple[str, float] | None:
    """Find the expected metric in task results.

    Uses the explicit metric from the segmentation CSV — no fallback.
    lm-eval-harness may use comma-suffixed keys like 'acc,none'.
    """
    for key, val in task_results.items():
        clean = key.replace(",none", "").split(",")[0].strip()
        if clean == expected_metric:
            return expected_metric, float(val)
    log.error(
        "Expected metric '%s' not found for %s. Available: %s",
        expected_metric, lm_eval_task, list(task_results.keys()),
    )
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
    missing: list[str] = []

    for _, task_row in seg.iterrows():
        task_name = task_row["task"]
        lm_eval = task_row["lm_eval_task"]
        cat = task_row["categoria_paper"]
        nt = task_row["native_translated"]
        expected_metric = task_row["metric"]

        if not lm_eval or lm_eval not in results:
            missing.append(task_name)
            continue

        task_results = results[lm_eval]
        resolved = _resolve_metric(task_results, lm_eval, expected_metric)
        if resolved is None:
            missing.append(task_name)
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

    if missing:
        log.error(
            "[%s] MISSING %d tasks (not in results or metric not found): %s",
            checkpoint_name, len(missing), ", ".join(missing),
        )
        sys.exit(1)

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
        "n_missing": 0,
        "all_score": round(all_score, 4),
        "native_score": round(native_score, 4) if native_score is not None else None,
        "translated_score": round(translated_score, 4) if translated_score is not None else None,
        "n_native": int(native_mask.sum()),
        "n_translated": int(translated_mask.sum()),
        "category_scores": {k: round(v, 4) for k, v in sorted(cat_scores.items())},
        "missing_tasks": [],
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
        default=DEFAULT_CHECKPOINT_ORDER,
        help="Ordered list of checkpoint names for processing and delta computation",
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
        sys.exit(1)

    # Discover result files
    order_index = {name: idx for idx, name in enumerate(args.checkpoint_order)}
    result_files = sorted(
        results_dir.glob("*.json"),
        key=lambda path: (order_index.get(path.stem, len(order_index)), path.stem.lower()),
    )
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
    order = [cp for cp in args.checkpoint_order if cp in checkpoint_names]
    order.extend(cp for cp in checkpoint_names if cp not in order)
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


if __name__ == "__main__":
    main()
