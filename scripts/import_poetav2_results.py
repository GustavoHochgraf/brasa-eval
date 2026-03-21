#!/usr/bin/env python
"""Import PoETa v2 evaluation results from the TuQwen-eval-PoETaV2 repo.

Reads individual task JSON files from the cloned repo and converts them
to the combined format expected by generate_scorecards.py.

Handles task name mapping between PoETaV2 naming and brasa-eval naming.

Usage:
    python scripts/import_poetav2_results.py --repo-dir ../TuQwen-eval-PoETaV2
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Task name mapping: PoETaV2 name -> brasa-eval lm_eval_task name
# Only tasks that differ need to be listed here
TASK_NAME_MAP = {
    "bluex": "bluex_greedy",
    "enem_2022_greedy": "enem2022_greedy",
    "faquad": "faquad_nli_greedy",
    "massive_greedy": "massive_pt_greedy",
    "mkqa_greedy": "mkqa_pt_greedy",
}

# Checkpoints to import: (result_dir_in_repo, output_filename, display_name)
CHECKPOINTS = [
    (
        "results/Qwen3-Base-results",
        "Qwen3-1.7B-Base",
        "Qwen/Qwen3-1.7B-Base",
    ),
    (
        "results/TuQwen3-Base-LR1e5-run1-1ep-results",
        "TuQwen3-Base-LR1e5-run1",
        "ggg-llms-team/TuQwen3-Base-LR1e5-run1",
    ),
    (
        "results/qwenrolina/LR1e5-b32g2gc8-order/QwenRolina3-Base-LR1e5-b32g2gc8-order-domain-1ep-results",
        "QwenRolina3-Base",
        "g4me/QwenRolina3-Base-LR1e5-b32g2gc8-order-domain",
    ),
]

# Metric preference order (same as generate_scorecards.py)
METRIC_PREFERENCE = ["acc_norm", "acc", "f1_macro", "f1-macro", "f1", "exact_match", "pearson"]


def git_show(repo_dir: Path, blob_path: str) -> str | None:
    """Read a file from git objects using 'git show'."""
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:{blob_path}"],
            capture_output=True, text=True, cwd=str(repo_dir),
        )
        if result.returncode == 0:
            return result.stdout
    except Exception:
        pass
    return None


def git_ls_tree(repo_dir: Path, tree_path: str) -> list[str]:
    """List files in a git tree."""
    result = subprocess.run(
        ["git", "ls-tree", "--name-only", "HEAD", tree_path + "/"],
        capture_output=True, text=True, cwd=str(repo_dir),
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]


def extract_task_result(task_data: dict) -> dict[str, float] | None:
    """Extract metrics from a PoETaV2 task result JSON.

    PoETaV2 format:
        {"results": {"task_name": {"dynamic-random": {"acc": 0.72, ...}}}}

    Returns flat dict like {"acc": 0.72} with the best metric.
    """
    results = task_data.get("results", {})
    for task_name, task_metrics in results.items():
        if isinstance(task_metrics, dict):
            # Handle nested "dynamic-random" key
            if "dynamic-random" in task_metrics:
                raw = task_metrics["dynamic-random"]
            else:
                raw = task_metrics

            clean: dict[str, float] = {}
            for key, val in raw.items():
                if "stderr" in key or "num_examples" in key:
                    continue
                try:
                    v = float(val)
                except (ValueError, TypeError):
                    continue
                # Normalize 0-100 scale to 0-1 for score metrics
                # (some PoETaV2 tasks report percentages instead of proportions)
                # Skip non-score keys like num_examples, thresholds, mse
                skip_normalize = {"num_examples", "best_f1_threshold", "mse", "unknown_pred"}
                if key not in skip_normalize and v > 1.0:
                    v = v / 100.0
                clean[key] = v

            if clean:
                return clean
    return None


def import_checkpoint(
    repo_dir: Path,
    result_dir: str,
    output_name: str,
    hf_model_id: str,
    output_dir: Path,
) -> None:
    """Import all task results for one checkpoint."""
    # List task JSON files
    all_files = git_ls_tree(repo_dir, result_dir)
    task_files = [
        f for f in all_files
        if f.endswith(".json") and "npm" not in f.lower() and "samples" not in f.lower()
    ]

    if not task_files:
        log.error("No task JSON files found in %s", result_dir)
        return

    combined_results: dict[str, dict[str, float]] = {}
    for fpath in task_files:
        content = git_show(repo_dir, fpath)
        if content is None:
            log.warning("Could not read: %s", fpath)
            continue

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            log.warning("Invalid JSON: %s", fpath)
            continue

        # Get original task name from the JSON results key
        results_dict = data.get("results", {})
        for orig_task_name in results_dict:
            metrics = extract_task_result(data)
            if metrics is None:
                log.warning("No usable metrics in %s", fpath)
                continue

            # Map task name if needed
            mapped_name = TASK_NAME_MAP.get(orig_task_name, orig_task_name)
            combined_results[mapped_name] = metrics
            break  # each file has one task

    # Write combined result file
    output = {
        "results": combined_results,
        "config": {
            "model": hf_model_id,
            "source": "TuQwen-eval-PoETaV2",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{output_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log.info("Wrote %s (%d tasks) from %s", out_path, len(combined_results), result_dir)

    # Report task name mapping applied
    mapped = {k: v for k, v in TASK_NAME_MAP.items() if v in combined_results}
    if mapped:
        log.info("  Task name mappings applied: %s", mapped)

    # Report missing tasks vs segmentation
    try:
        import pandas as pd
        seg_path = REPO_ROOT / "data" / "paper_final_segmentation.csv"
        if seg_path.exists():
            seg = pd.read_csv(seg_path)
            expected = set(seg["lm_eval_task"].dropna())
            found = set(combined_results.keys())
            missing = expected - found
            if missing:
                log.info("  Missing %d tasks vs segmentation: %s", len(missing), sorted(missing))
            extra = found - expected
            if extra:
                log.info("  Extra %d tasks not in segmentation: %s", len(extra), sorted(extra))
    except ImportError:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Import PoETa v2 results from TuQwen-eval-PoETaV2 repo")
    p.add_argument("--repo-dir", default=str(REPO_ROOT.parent / "TuQwen-eval-PoETaV2"),
                    help="Path to cloned TuQwen-eval-PoETaV2 repo")
    p.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "eval_results"),
                    help="Output directory for converted result JSONs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.repo_dir)
    output_dir = Path(args.output_dir)

    if not (repo_dir / ".git").exists():
        log.error("Not a git repo: %s", repo_dir)
        return

    log.info("Importing from: %s", repo_dir)
    log.info("Output dir: %s", output_dir)
    log.info("")

    for result_dir, output_name, hf_model_id in CHECKPOINTS:
        log.info("--- %s ---", output_name)
        import_checkpoint(repo_dir, result_dir, output_name, hf_model_id, output_dir)
        log.info("")

    log.info("Done. Run 'python scripts/generate_scorecards.py --results_dir %s' to generate scorecards.", output_dir)


if __name__ == "__main__":
    main()
