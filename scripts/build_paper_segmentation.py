#!/usr/bin/env python
"""D1 — Build the canonical paper segmentation table for PoETa v2.

Reads the TaskMap Excel (44 tasks from Table 3) and the Tucano/PT overlap
spreadsheet, then produces a deterministic CSV with paper-oriented columns.

4 tasks are excluded because their datasets are private to Maritaca AI:
POSComp, ARC Challenge, ARC Easy, Ethics Commonsense.

Final output: 40 tasks (14 Native + 26 Translated).

Usage:
    python scripts/build_paper_segmentation.py
    python scripts/build_paper_segmentation.py --taskmap path/to/TaskMap.xlsx --overlap path/to/overlap.xlsx
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tasks excluded from the benchmark (private Maritaca AI datasets)
# ---------------------------------------------------------------------------
EXCLUDED_TASKS = {"POSComp", "ARC Challenge", "ARC Easy", "Ethics Commonsense"}

# ---------------------------------------------------------------------------
# Explicit metric per task (must match what lm-eval-harness actually reports)
# ---------------------------------------------------------------------------
TASK_METRIC: dict[str, str] = {
    "Assin STS": "pearson",
    "BLUEX": "acc_norm",
    "Faquad": "f1",
    "MKQA": "best_em",
}
DEFAULT_METRIC = "acc"

# ---------------------------------------------------------------------------
# Canonical paper category mapping (deterministic, one per task)
# Categories: Brazil / exams / culture | Social / safety |
#             Text understanding / QA / classification | Reasoning | Math | Code / other
# ---------------------------------------------------------------------------
CATEGORIA_PAPER: dict[str, str] = {
    # --- Native: Brazil / exams / culture ---
    "BLUEX": "Brazil / exams / culture",
    "Enem": "Brazil / exams / culture",
    "Enem 2022": "Brazil / exams / culture",
    "Broverbs History to Proverb": "Brazil / exams / culture",
    "Broverbs Proverb to History": "Brazil / exams / culture",
    "Repro": "Brazil / exams / culture",
    # POSComp excluded (private Maritaca AI dataset)
    # --- Native: Social / safety ---
    "TweetsentBR": "Social / safety",
    "Mina BR": "Social / safety",
    "PT Hate Speech": "Social / safety",
    "HateBR Binary": "Social / safety",
    # --- Native: Text understanding / QA / classification ---
    "Assin RTE": "Text understanding / QA / classification",
    "Assin STS": "Text understanding / QA / classification",
    "Faquad": "Text understanding / QA / classification",
    # --- Native: Reasoning ---
    "InferBR": "Reasoning",
    # --- Translated: Text understanding / QA / classification ---
    "AGNews": "Text understanding / QA / classification",
    "BoolQ": "Text understanding / QA / classification",
    "IMDb": "Text understanding / QA / classification",
    "Massive": "Text understanding / QA / classification",
    "MKQA": "Text understanding / QA / classification",
    "SST-2": "Text understanding / QA / classification",
    "BB General Knowledge": "Text understanding / QA / classification",
    "BB VitaminC Fact Verification": "Text understanding / QA / classification",
    # --- Translated: Social / safety ---
    "BB Simple Ethical Questions": "Social / safety",
    "BB BBQ": "Social / safety",
    # Ethics Commonsense excluded (private Maritaca AI dataset)
    # --- Translated: Reasoning ---
    "WSC-285": "Reasoning",
    "StoryCloze": "Reasoning",
    "BB Social IQA": "Reasoning",
    "BB Analogical Similarity": "Reasoning",
    "BB Empirical Judgments": "Reasoning",
    "BB Fallacies Syllogisms": "Reasoning",
    "BB StrategyQA": "Reasoning",
    "BB Causal Judgment": "Reasoning",
    "BB Cause and Effect": "Reasoning",
    # ARC Challenge, ARC Easy excluded (private Maritaca AI datasets)
    "Balanced COPA": "Reasoning",
    "LogiQA": "Reasoning",
    # --- Translated: Math ---
    "BB Mathematical Induction": "Math",
    "Math MC": "Math",
    "GSM8K MC": "Math",
    "AGIEval SAT Math": "Math",
    # --- Translated: Code / other ---
    "BB Code Line Description": "Code / other",
}

# ---------------------------------------------------------------------------
# lm_eval_task names (from task_map.py manual mapping + fuzzy inference)
# ---------------------------------------------------------------------------
LM_EVAL_TASK: dict[str, str] = {
    "Assin RTE": "assin_rte_greedy",
    "Assin STS": "assin_sts_greedy",
    "BLUEX": "bluex_greedy",
    "Enem": "enem_greedy",
    "Enem 2022": "enem2022_greedy",
    "Faquad": "faquad_nli_greedy",
    "TweetsentBR": "tweetsentbr_greedy",
    "Broverbs History to Proverb": "broverbs_history_to_proverb_greedy",
    "Broverbs Proverb to History": "broverbs_proverb_to_history_greedy",
    "InferBR": "inferbr_greedy",
    "Repro": "repro_greedy",
    "Mina BR": "mina_br_greedy",
    "PT Hate Speech": "pt_hate_speech_greedy",
    "HateBR Binary": "hatebr_binary_greedy",
    # POSComp excluded
    "AGNews": "agnews_pt_greedy",
    "BoolQ": "boolq_pt_greedy",
    "IMDb": "imdb_pt_greedy",
    "Massive": "massive_pt_greedy",
    "MKQA": "mkqa_pt_greedy",
    "SST-2": "sst2_pt_greedy",
    "WSC-285": "wsc285_pt_greedy",
    "StoryCloze": "storycloze_pt_greedy",
    # ARC Challenge, ARC Easy, Ethics Commonsense excluded
    "Math MC": "math_mc_greedy",
    "GSM8K MC": "gsm8k_mc_greedy",
    "AGIEval SAT Math": "agieval_sat_math_greedy",
    "Balanced COPA": "balanced_copa_greedy",
    "LogiQA": "logiqa_greedy",
    "BB Analogical Similarity": "bigbench_pt_analogical_similarity_greedy",
    "BB Code Line Description": "bigbench_pt_code_line_description_greedy",
    "BB Empirical Judgments": "bigbench_pt_empirical_judgments_greedy",
    "BB Fallacies Syllogisms": "bigbench_pt_formal_fallacies_syllogisms_negation_greedy",
    "BB General Knowledge": "bigbench_pt_general_knowledge_greedy",
    "BB Mathematical Induction": "bigbench_pt_mathematical_induction_greedy",
    "BB Simple Ethical Questions": "bigbench_pt_simple_ethical_questions_greedy",
    "BB StrategyQA": "bigbench_pt_strategyqa_greedy",
    "BB VitaminC Fact Verification": "bigbench_pt_vitaminc_fact_verification_greedy",
    "BB Social IQA": "bigbench_pt_social_iqa_greedy",
    "BB Causal Judgment": "bigbench_pt_causal_judgment_greedy",
    "BB BBQ": "bigbench_pt_bbq_greedy",
    "BB Cause and Effect": "bigbench_pt_cause_and_effect_two_sentences_greedy",
}

# ---------------------------------------------------------------------------
# Benchmark origin (source dataset family)
# ---------------------------------------------------------------------------
BENCHMARK_ORIGEM: dict[str, str] = {
    "Assin RTE": "ASSIN2",
    "Assin STS": "ASSIN2",
    "BLUEX": "BLUEX",
    "Enem": "ENEM",
    "Enem 2022": "ENEM",
    "Faquad": "FaQuAD",
    "TweetsentBR": "TweetSentBR",
    "Broverbs History to Proverb": "BRoverbs",
    "Broverbs Proverb to History": "BRoverbs",
    "InferBR": "InferBR",
    "Repro": "Repro",
    "Mina BR": "Mina",
    "PT Hate Speech": "PT Hate Speech",
    "HateBR Binary": "HateBR",
    # POSComp excluded
    "AGNews": "AG News",
    "BoolQ": "BoolQ",
    "IMDb": "IMDb",
    "Massive": "MASSIVE",
    "MKQA": "MKQA",
    "SST-2": "SST-2",
    "WSC-285": "WSC-285",
    "StoryCloze": "StoryCloze",
    # ARC Challenge, ARC Easy, Ethics Commonsense excluded
    "Math MC": "MATH",
    "GSM8K MC": "GSM8K",
    "AGIEval SAT Math": "AGIEval",
    "Balanced COPA": "COPA",
    "LogiQA": "LogiQA",
}
# All BB tasks share origin
for k in CATEGORIA_PAPER:
    if k.startswith("BB "):
        BENCHMARK_ORIGEM.setdefault(k, "BIG-Bench")


def _build_overlap_index(overlap_path: Path) -> dict[str, dict]:
    """Build lookup from overlap spreadsheet keyed by normalized task name."""
    if not overlap_path.exists():
        log.warning("Overlap spreadsheet not found: %s", overlap_path)
        return {}
    df = pd.read_excel(overlap_path)
    df.columns = [c.replace("\n", " ").strip() for c in df.columns]

    index: dict[str, dict] = {}
    for _, row in df.iterrows():
        task = str(row.get("Task / Dataset", "")).strip()
        if not task or task.startswith("SHARED") or task.startswith("PoETa") or task.startswith("TUCANO") or task.startswith("TRANSLATED"):
            continue
        # suite presence
        suites = []
        if str(row.get("PoETa v1 (14 tasks)", "")).strip() == "✓":
            suites.append("PoETa-v1")
        if str(row.get("PoETa v2 (44 tasks)", "")).strip() == "✓":
            suites.append("PoETa-v2")
        if str(row.get("PT Leaderboard (9 tasks)", "")).strip() == "✓":
            suites.append("PT-Leaderboard")
        tucano_val = str(row.get("Tucano Harness (14+ tasks)", "")).strip()
        if "✓" in tucano_val:
            suites.append("Tucano")

        overlap_class = str(row.get("Overlap Class", "")).strip()
        comparability = str(row.get("Comparability", "")).strip()

        index[task] = {
            "suite_presence": "; ".join(suites) if suites else "PoETa-v2",
            "overlap_class": overlap_class if overlap_class and overlap_class != "nan" else "Non-overlap",
            "comparability": comparability if comparability and comparability != "nan" else "High",
        }
    return index


# Fuzzy name matching from TaskMap names to overlap sheet names
TASKMAP_TO_OVERLAP: dict[str, str] = {
    "Assin RTE": "ASSIN2 RTE",
    "Assin STS": "ASSIN2 STS",
    "BLUEX": "BLUEX",
    "Enem": "ENEM Challenge",
    "Enem 2022": "ENEM 2022",
    "Faquad": "FaQuAD",
    "TweetsentBR": "TweetSentBr",
    "Broverbs History to Proverb": "BRoverbs",
    "Broverbs Proverb to History": "BRoverbs",
    "InferBR": "InferBR",
    "PT Hate Speech": "PT Hate Speech",
    "HateBR Binary": "HateBR",
    "AGNews": "AG News-PT",
    "BoolQ": "BoolQ-PT",
    "IMDb": "IMDB-PT",
    "Massive": "MASSIVE-PT",
    "MKQA": "MKQA-PT",
    "SST-2": "SST2-PT",
    "WSC-285": "WSC285-PT",
    "BB Causal Judgment": "BB: Causal Judgment",
    "LogiQA": "LogiQA-PT",
    # ARC Challenge excluded
}


def _normalize_overlap_class(raw: str) -> str:
    raw = raw.strip()
    if "strict" in raw.lower():
        return "Strict"
    if "partial" in raw.lower():
        return "Partial"
    return "Non-overlap"


def _normalize_comparability(raw: str) -> str:
    raw = raw.strip()
    if "high" in raw.lower():
        return "High"
    if "medium" in raw.lower():
        return "Medium"
    if "low" in raw.lower():
        return "Low"
    return "High"


def build_segmentation(
    taskmap_path: Path,
    overlap_path: Path,
) -> pd.DataFrame:
    """Build the canonical paper segmentation from source spreadsheets."""
    # --- Load TaskMap (44 tasks) ---
    tm = pd.read_excel(taskmap_path)
    log.info("TaskMap loaded: %d tasks", len(tm))
    assert len(tm) == 44, f"Expected 44 tasks, got {len(tm)}"

    # --- Exclude private Maritaca AI datasets ---
    tm = tm[~tm["Dataset"].str.strip().isin(EXCLUDED_TASKS)].copy()
    log.info("After excluding %d private tasks: %d tasks remain", len(EXCLUDED_TASKS), len(tm))

    # --- Load overlap index ---
    overlap_idx = _build_overlap_index(overlap_path)
    log.info("Overlap index: %d entries", len(overlap_idx))

    rows: list[dict] = []
    for _, r in tm.iterrows():
        task = str(r["Dataset"]).strip()
        # Rename MKOA -> MKQA (typo in original TaskMap)
        if task == "MKOA":
            task = "MKQA"
        few_shot = int(r["Few-Shot"])
        translated_raw = str(r["Translated"]).strip()
        native_translated = "Translated" if translated_raw.lower() in ("yes", "true", "1", "sim") else "Native"
        original_subcategories = str(r["Categories"]).strip()

        # Paper category
        cat = CATEGORIA_PAPER.get(task)
        if cat is None:
            log.warning("No categoria_paper for task: %s — defaulting to 'Text understanding / QA / classification'", task)
            cat = "Text understanding / QA / classification"

        # Benchmark origin
        origem = BENCHMARK_ORIGEM.get(task, task)

        # lm_eval_task
        lm_eval = LM_EVAL_TASK.get(task, "")

        # Overlap data
        overlap_name = TASKMAP_TO_OVERLAP.get(task)
        overlap_data = overlap_idx.get(overlap_name, {}) if overlap_name else {}

        overlap_class = _normalize_overlap_class(overlap_data.get("overlap_class", "Non-overlap"))
        comparability = _normalize_comparability(overlap_data.get("comparability", "High"))
        suite_presence = overlap_data.get("suite_presence", "PoETa-v2")

        # Ensure PoETa-v2 is always in suite_presence
        if "PoETa-v2" not in suite_presence:
            suite_presence = "PoETa-v2; " + suite_presence if suite_presence else "PoETa-v2"

        # Notes
        notes = ""
        if task.startswith("BB "):
            notes = "BIG-Bench translated task"
        if native_translated == "Native" and "brazil" in original_subcategories.lower():
            notes = "Brazilian-origin dataset"
        if task in ("Massive", "MKQA"):
            notes = "Natively multilingual; classified as Translated following PoETa v2 protocol"

        # Explicit metric per task
        metric = TASK_METRIC.get(task, DEFAULT_METRIC)

        rows.append({
            "task": task,
            "lm_eval_task": lm_eval,
            "metric": metric,
            "benchmark_origem": origem,
            "categoria_paper": cat,
            "native_translated": native_translated,
            "overlap_class": overlap_class,
            "comparability": comparability,
            "few_shot": few_shot,
            "original_subcategories": original_subcategories,
            "suite_presence": suite_presence,
            "notes_curtas": notes,
        })

    df = pd.DataFrame(rows)

    # --- Validation ---
    assert len(df) == 40, f"Expected 40 tasks after exclusion, got {len(df)}"

    n_native = (df["native_translated"] == "Native").sum()
    n_translated = (df["native_translated"] == "Translated").sum()
    assert n_native == 14, f"Expected 14 Native tasks, got {n_native}"
    assert n_translated == 26, f"Expected 26 Translated tasks, got {n_translated}"

    assert set(df["native_translated"]) <= {"Native", "Translated"}
    assert set(df["overlap_class"]) <= {"Strict", "Partial", "Non-overlap"}
    assert set(df["comparability"]) <= {"High", "Medium", "Low"}
    expected_cats = {
        "Brazil / exams / culture",
        "Social / safety",
        "Text understanding / QA / classification",
        "Reasoning",
        "Math",
        "Code / other",
    }
    actual_cats = set(df["categoria_paper"])
    assert actual_cats <= expected_cats, f"Unexpected categories: {actual_cats - expected_cats}"
    assert set(df["metric"]) <= {"acc", "acc_norm", "pearson", "f1", "best_em"}, \
        f"Unexpected metrics: {set(df['metric'])}"

    log.info("Segmentation built: %d tasks (14 Native + 26 Translated)", len(df))
    log.info("  Categories: %s", df["categoria_paper"].value_counts().to_dict())
    log.info("  Metrics: %s", df["metric"].value_counts().to_dict())

    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build paper segmentation table for PoETa v2")
    p.add_argument(
        "--taskmap",
        default="C:/Users/gusth/Mestrado/projeto/PoetaV2-TaskMap.xlsx",
        help="Path to PoETa v2 TaskMap Excel",
    )
    p.add_argument(
        "--overlap",
        default="C:/Users/gusth/Mestrado/projeto/poeta_tucano_comparison_overlap_classified.xlsx",
        help="Path to Tucano/PT overlap comparison Excel",
    )
    p.add_argument("--out", default="data/paper_final_segmentation.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    taskmap_path = Path(args.taskmap)
    overlap_path = Path(args.overlap)
    out_path = Path(args.out)

    if not taskmap_path.exists():
        log.error("TaskMap not found: %s", taskmap_path)
        sys.exit(1)

    df = build_segmentation(taskmap_path, overlap_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Wrote: %s", out_path)


if __name__ == "__main__":
    main()
