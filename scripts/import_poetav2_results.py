#!/usr/bin/env python
"""Import PoETa v2 evaluation results from a private source repository.

Reads task-level JSON files from a cloned evaluation repository and converts
them to the combined format expected by `generate_scorecards.py`.

The importer auto-detects the three paper checkpoints:
  - Qwen 1.7B Base
  - Gigaverbo adapted
  - Carolina adapted

Usage:
    python scripts/import_poetav2_results.py
    python scripts/import_poetav2_results.py --repo-dir ../poeta_v2_eval_private
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_DIR = REPO_ROOT.parent / "poeta_v2_eval_private"
MODEL_ID_RE = re.compile(r"pretrained=([^,\s]+)")

# Task name mapping: source repo name -> brasa-eval lm_eval_task name.
TASK_NAME_MAP = {
    "bluex": "bluex_greedy",
    "enem_2022_greedy": "enem2022_greedy",
    "faquad": "faquad_nli_greedy",
    "massive_greedy": "massive_pt_greedy",
    "mkqa_greedy": "mkqa_pt_greedy",
}

CHECKPOINT_SPECS = [
    {
        "name": "Qwen 1.7B Base",
        "owner": "Qwen",
        "required_tokens": ["base-results"],
        "preferred_tokens": ["qwen3"],
        "forbidden_tokens": ["instruct"],
    },
    {
        "name": "Gigaverbo adapted",
        "owner": "ggg-llms-team",
        "required_tokens": ["1ep"],
        "preferred_tokens": ["lr1e5", "run1"],
        "forbidden_tokens": ["instruct", "run2", "lr8e5"],
    },
    {
        "name": "Carolina adapted",
        "owner": "g4me",
        "required_tokens": ["1ep"],
        "preferred_tokens": ["lr1e5", "order", "domain", "b32g2gc8"],
        "forbidden_tokens": ["wsd", "2ep", "3ep", "mix", "ppl", "irm", "b64g8"],
    },
]


def git_show(repo_dir: Path, blob_path: str) -> str | None:
    """Read a file from git objects using `git show`."""
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:{blob_path}"],
            capture_output=True,
            text=True,
            cwd=str(repo_dir),
            check=False,
        )
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def git_ls_tree(repo_dir: Path, tree_path: str, *, recursive: bool = False, directories: bool = False) -> list[str]:
    """List files or directories in a git tree."""
    cmd = ["git", "ls-tree"]
    if directories:
        cmd.append("-d")
    if recursive:
        cmd.append("-r")
    cmd.extend(["--name-only", "HEAD", tree_path])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo_dir),
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def extract_task_result(task_data: dict) -> dict[str, float] | None:
    """Extract metrics from a task result JSON."""
    results = task_data.get("results", {})
    for _, task_metrics in results.items():
        if not isinstance(task_metrics, dict):
            continue

        raw = task_metrics.get("dynamic-random", task_metrics)
        clean: dict[str, float] = {}
        for key, val in raw.items():
            if "stderr" in key or "num_examples" in key:
                continue
            try:
                value = float(val)
            except (TypeError, ValueError):
                continue

            skip_normalize = {"num_examples", "best_f1_threshold", "mse", "unknown_pred"}
            if key not in skip_normalize and value > 1.0:
                value = value / 100.0
            clean[key] = value

        if clean:
            return clean
    return None


def _extract_model_id(blob_content: str) -> str | None:
    match = MODEL_ID_RE.search(blob_content)
    if not match:
        return None
    return match.group(1)


def _result_task_files(repo_dir: Path, result_dir: str) -> list[str]:
    files = git_ls_tree(repo_dir, f"{result_dir}/", recursive=False)
    return [
        path for path in files
        if path.endswith(".json") and "npm" not in path.lower() and "samples" not in path.lower()
    ]


def _score_candidate(result_dir: str, spec: dict[str, object]) -> int | None:
    path_lower = result_dir.lower()
    required_tokens = spec.get("required_tokens", [])
    preferred_tokens = spec.get("preferred_tokens", [])
    forbidden_tokens = spec.get("forbidden_tokens", [])

    if any(token not in path_lower for token in required_tokens):
        return None

    score = 0
    for token in preferred_tokens:
        if token in path_lower:
            score += 3
    for token in forbidden_tokens:
        if token in path_lower:
            score -= 4

    # Prefer shallower directories when scores tie.
    score -= result_dir.count("/")
    return score


def discover_checkpoints(repo_dir: Path) -> list[tuple[str, str]]:
    """Auto-discover the three paper checkpoints inside the source repo."""
    result_dirs = [
        path for path in git_ls_tree(repo_dir, "results", recursive=True, directories=True)
        if path.endswith("-results")
    ]

    choices: dict[str, tuple[int, str]] = {}
    for result_dir in result_dirs:
        task_files = _result_task_files(repo_dir, result_dir)
        if len(task_files) < 30:
            continue

        sample_blob = git_show(repo_dir, task_files[0])
        if sample_blob is None:
            continue
        model_id = _extract_model_id(sample_blob)
        if not model_id or "/" not in model_id:
            continue

        owner = model_id.split("/", 1)[0]
        for spec in CHECKPOINT_SPECS:
            if owner != spec["owner"]:
                continue
            score = _score_candidate(result_dir, spec)
            if score is None:
                continue
            current = choices.get(spec["name"])
            if current is None or score > current[0]:
                choices[spec["name"]] = (score, result_dir)

    missing = [spec["name"] for spec in CHECKPOINT_SPECS if spec["name"] not in choices]
    if missing:
        available = ", ".join(sorted(result_dirs))
        raise RuntimeError(
            f"Could not auto-discover checkpoint(s): {', '.join(missing)}. "
            f"Available result directories: {available}"
        )

    ordered: list[tuple[str, str]] = []
    for spec in CHECKPOINT_SPECS:
        _, result_dir = choices[spec["name"]]
        ordered.append((spec["name"], result_dir))
    return ordered


def import_checkpoint(
    repo_dir: Path,
    result_dir: str,
    output_name: str,
    output_dir: Path,
) -> None:
    """Import all task results for one checkpoint."""
    task_files = _result_task_files(repo_dir, result_dir)
    if not task_files:
        log.error("No task JSON files found in %s", result_dir)
        return

    combined_results: dict[str, dict[str, float]] = {}
    for file_path in task_files:
        content = git_show(repo_dir, file_path)
        if content is None:
            log.warning("Could not read: %s", file_path)
            continue

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            log.warning("Invalid JSON: %s", file_path)
            continue

        results_dict = data.get("results", {})
        for source_task_name in results_dict:
            metrics = extract_task_result(data)
            if metrics is None:
                log.warning("No usable metrics in %s", file_path)
                continue
            mapped_name = TASK_NAME_MAP.get(source_task_name, source_task_name)
            combined_results[mapped_name] = metrics
            break

    output = {
        "results": combined_results,
        "config": {
            "model": output_name,
            "source": "private PoETa v2 evaluation repo",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{output_name}.json"
    with open(out_path, "w", encoding="utf-8") as file_handle:
        json.dump(output, file_handle, indent=2, ensure_ascii=False)

    log.info("Wrote %s (%d tasks) from %s", out_path, len(combined_results), result_dir)


def resolve_repo_dir(requested_repo_dir: Path) -> Path | None:
    """Resolve a reasonable default source repo path."""
    if (requested_repo_dir / ".git").exists():
        return requested_repo_dir
    if (DEFAULT_REPO_DIR / ".git").exists():
        return DEFAULT_REPO_DIR

    for sibling in sorted(REPO_ROOT.parent.iterdir()):
        if not sibling.is_dir():
            continue
        if not (sibling / ".git").exists():
            continue
        if not (sibling / "results" / "Qwen3-Base-results").exists():
            continue
        return sibling
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import PoETa v2 results into brasa-eval format")
    parser.add_argument(
        "--repo-dir",
        default=str(DEFAULT_REPO_DIR),
        help="Path to the cloned private PoETa v2 evaluation repository",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "eval_results"),
        help="Output directory for converted result JSONs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_dir = resolve_repo_dir(Path(args.repo_dir))
    output_dir = Path(args.output_dir)

    if repo_dir is None:
        log.error(
            "Could not locate the source results repository. "
            "Pass it explicitly with --repo-dir."
        )
        return

    log.info("Importing from: %s", repo_dir)
    log.info("Output dir: %s", output_dir)
    log.info("")

    try:
        checkpoint_dirs = discover_checkpoints(repo_dir)
    except RuntimeError as exc:
        log.error("%s", exc)
        return

    for output_name, result_dir in checkpoint_dirs:
        log.info("--- %s ---", output_name)
        import_checkpoint(repo_dir, result_dir, output_name, output_dir)
        log.info("")

    log.info(
        "Done. Run 'python scripts/generate_scorecards.py --results_dir %s' to generate scorecards.",
        output_dir,
    )


if __name__ == "__main__":
    main()
