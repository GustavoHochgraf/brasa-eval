#!/usr/bin/env python
"""D1 — Build the canonical paper segmentation table for PoETa v2.

Reads the TaskMap Excel (44 tasks from Table 3) and the Tucano/PT overlap
spreadsheet, then produces a deterministic CSV with paper-oriented columns.

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
# Canonical paper category mapping (deterministic, one per task)
# Categories: Brazil / exams / culture | toxicity / social |
#             NLI / text understanding | reasoning | math | code / other
# ---------------------------------------------------------------------------
CATEGORIA_PAPER: dict[str, str] = {
    # --- Native: Brazil / exams / culture ---
    "BLUEX": "Brazil / exams / culture",
    "Enem": "Brazil / exams / culture",
    "Enem 2022": "Brazil / exams / culture",
    "Broverbs History to Proverb": "Brazil / exams / culture",
    "Broverbs Proverb to History": "Brazil / exams / culture",
    "Repro": "Brazil / exams / culture",
    "POSComp": "Brazil / exams / culture",
    # --- Native: toxicity / social ---
    "TweetsentBR": "toxicity / social",
    "Mina BR": "toxicity / social",
    "PT Hate Speech": "toxicity / social",
    "HateBR Binary": "toxicity / social",
    # --- Native: NLI / text understanding ---
    "Assin RTE": "NLI / text understanding",
    "Assin STS": "NLI / text understanding",
    "Faquad": "NLI / text understanding",
    "InferBR": "NLI / text understanding",
    # --- Translated: NLI / text understanding ---
    "AGNews": "NLI / text understanding",
    "BoolQ": "NLI / text understanding",
    "IMDb": "NLI / text understanding",
    "Massive": "NLI / text understanding",
    "MKOA": "NLI / text understanding",
    "SST-2": "NLI / text understanding",
    "WSC-285": "NLI / text understanding",
    "StoryCloze": "NLI / text understanding",
    "BB VitaminC Fact Verification": "NLI / text understanding",
    "BB General Knowledge": "NLI / text understanding",
    "BB Social IQA": "NLI / text understanding",
    # --- Translated: toxicity / social ---
    "BB Simple Ethical Questions": "toxicity / social",
    "BB BBQ": "toxicity / social",
    "Ethics Commonsense": "toxicity / social",
    # --- Translated: reasoning ---
    "BB Analogical Similarity": "reasoning",
    "BB Empirical Judgments": "reasoning",
    "BB Fallacies Syllogisms": "reasoning",
    "BB StrategyQA": "reasoning",
    "BB Causal Judgment": "reasoning",
    "BB Cause and Effect": "reasoning",
    "ARC Challenge": "reasoning",
    "ARC Easy": "reasoning",
    "Balanced COPA": "reasoning",
    "LogiQA": "reasoning",
    # --- Translated: math ---
    "BB Mathematical Induction": "math",
    "Math MC": "math",
    "GSM8K MC": "math",
    "AGIEval SAT Math": "math",
    # --- Translated: code / other ---
    "BB Code Line Description": "code / other",
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
    "POSComp": "poscomp_greedy",
    "AGNews": "agnews_pt_greedy",
    "BoolQ": "boolq_pt_greedy",
    "IMDb": "imdb_pt_greedy",
    "Massive": "massive_pt_greedy",
    "MKOA": "mkqa_pt_greedy",
    "SST-2": "sst2_pt_greedy",
    "WSC-285": "wsc285_pt_greedy",
    "StoryCloze": "storycloze_pt_greedy",
    "ARC Challenge": "arc_challenge_greedy_pt",
    "ARC Easy": "arc_easy_greedy_pt",
    "Ethics Commonsense": "ethics_commonsense_test_hard_greedy",
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
    "POSComp": "POSComp",
    "AGNews": "AG News",
    "BoolQ": "BoolQ",
    "IMDb": "IMDb",
    "Massive": "MASSIVE",
    "MKOA": "MKQA",
    "SST-2": "SST-2",
    "WSC-285": "WSC-285",
    "StoryCloze": "StoryCloze",
    "ARC Challenge": "ARC",
    "ARC Easy": "ARC",
    "Ethics Commonsense": "ETHICS",
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
    "MKOA": "MKQA-PT",
    "SST-2": "SST2-PT",
    "WSC-285": "WSC285-PT",
    "BB Causal Judgment": "BB: Causal Judgment",
    "LogiQA": "LogiQA-PT",
    "ARC Challenge": "ARC-Challenge-PT",
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

    # --- Load overlap index ---
    overlap_idx = _build_overlap_index(overlap_path)
    log.info("Overlap index: %d entries", len(overlap_idx))

    rows: list[dict] = []
    for _, r in tm.iterrows():
        task = str(r["Dataset"]).strip()
        few_shot = int(r["Few-Shot"])
        translated_raw = str(r["Translated"]).strip()
        native_translated = "Translated" if translated_raw.lower() in ("yes", "true", "1", "sim") else "Native"
        original_subcategories = str(r["Categories"]).strip()

        # Paper category
        cat = CATEGORIA_PAPER.get(task)
        if cat is None:
            log.warning("No categoria_paper for task: %s — defaulting to 'NLI / text understanding'", task)
            cat = "NLI / text understanding"

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
        if task in ("Massive", "MKOA"):
            notes = "Natively multilingual (not translated from EN)"

        rows.append({
            "task": task,
            "lm_eval_task": lm_eval,
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
    assert set(df["native_translated"]) <= {"Native", "Translated"}
    assert set(df["overlap_class"]) <= {"Strict", "Partial", "Non-overlap"}
    assert set(df["comparability"]) <= {"High", "Medium", "Low"}
    expected_cats = {
        "Brazil / exams / culture",
        "toxicity / social",
        "NLI / text understanding",
        "reasoning",
        "math",
        "code / other",
    }
    actual_cats = set(df["categoria_paper"])
    assert actual_cats <= expected_cats, f"Unexpected categories: {actual_cats - expected_cats}"

    log.info("Segmentation built: %d tasks", len(df))
    log.info("  Native: %d, Translated: %d", (df["native_translated"] == "Native").sum(), (df["native_translated"] == "Translated").sum())
    log.info("  Categories: %s", df["categoria_paper"].value_counts().to_dict())
    log.info("  Overlap: %s", df["overlap_class"].value_counts().to_dict())

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

    # Also write JSON for programmatic use
    out_json = out_path.with_suffix(".json")
    df.to_json(out_json, orient="records", indent=2, force_ascii=False)
    log.info("Wrote: %s", out_json)


if __name__ == "__main__":
    main()
