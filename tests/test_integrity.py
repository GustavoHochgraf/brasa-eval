"""Integrity tests for the brasa-eval analysis pipeline.

Verifies that the segmentation CSV, evaluation results, and generated
scorecards are internally consistent and match the paper's claims.

Run:
    pytest tests/test_integrity.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SEG_PATH = ROOT / "data" / "paper_final_segmentation.csv"
RESULTS_DIR = ROOT / "outputs" / "eval_results"
SCORECARDS_DIR = ROOT / "outputs" / "scorecards"

CHECKPOINTS = [
    "Qwen3-1.7B-Base",
    "TuQwen3-Base-LR1e5-run1",
    "QwenRolina3-Base",
]


@pytest.fixture
def seg() -> pd.DataFrame:
    return pd.read_csv(SEG_PATH)


@pytest.fixture
def eval_results() -> dict[str, dict]:
    results = {}
    for cp in CHECKPOINTS:
        path = RESULTS_DIR / f"{cp}.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results[cp] = data.get("results", data)
    return results


# ── Segmentation CSV structure ────────────────────────────────────────

class TestSegmentation:
    def test_exactly_40_tasks(self, seg: pd.DataFrame):
        assert len(seg) == 40, f"Expected 40 tasks, got {len(seg)}"

    def test_14_native_26_translated(self, seg: pd.DataFrame):
        native = (seg["native_translated"] == "Native").sum()
        translated = (seg["native_translated"] == "Translated").sum()
        assert native == 14, f"Expected 14 Native, got {native}"
        assert translated == 26, f"Expected 26 Translated, got {translated}"

    def test_category_counts(self, seg: pd.DataFrame):
        counts = seg["categoria_paper"].value_counts().to_dict()
        expected = {
            "NLI / text understanding": 15,
            "reasoning": 8,
            "Brazil / exams / culture": 6,
            "toxicity / social": 6,
            "math": 4,
            "code / other": 1,
        }
        assert counts == expected, f"Category counts mismatch: {counts} != {expected}"

    def test_no_excluded_tasks(self, seg: pd.DataFrame):
        excluded = {"POSComp", "ARC Challenge", "ARC Easy", "Ethics Commonsense"}
        present = set(seg["task"]) & excluded
        assert not present, f"Excluded tasks found in CSV: {present}"

    def test_no_mkoa_typo(self, seg: pd.DataFrame):
        assert "MKOA" not in seg["task"].values, "MKOA typo still present (should be MKQA)"

    def test_metric_column_exists(self, seg: pd.DataFrame):
        assert "metric" in seg.columns, "Missing 'metric' column"

    def test_explicit_metrics(self, seg: pd.DataFrame):
        special = {
            "Assin STS": "pearson",
            "BLUEX": "acc_norm",
            "Faquad": "f1",
            "MKQA": "best_em",
        }
        for task, expected_metric in special.items():
            row = seg[seg["task"] == task]
            assert not row.empty, f"Task {task} not found"
            actual = row.iloc[0]["metric"]
            assert actual == expected_metric, (
                f"{task}: expected metric '{expected_metric}', got '{actual}'"
            )

    def test_default_metric_is_acc(self, seg: pd.DataFrame):
        special_tasks = {"Assin STS", "BLUEX", "Faquad", "MKQA"}
        others = seg[~seg["task"].isin(special_tasks)]
        non_acc = others[others["metric"] != "acc"]
        assert non_acc.empty, (
            f"Tasks with non-acc metric (expected acc): "
            f"{dict(zip(non_acc['task'], non_acc['metric']))}"
        )

    def test_unique_lm_eval_tasks(self, seg: pd.DataFrame):
        dupes = seg[seg["lm_eval_task"].duplicated(keep=False)]
        assert dupes.empty, f"Duplicate lm_eval_task entries: {dupes['lm_eval_task'].tolist()}"

    def test_massive_mkqa_notes(self, seg: pd.DataFrame):
        for task in ["Massive", "MKQA"]:
            row = seg[seg["task"] == task]
            notes = row.iloc[0]["notes_curtas"]
            assert "Translated" in notes or "multilingual" in notes.lower(), (
                f"{task} should note multilingual classification"
            )


# ── Eval results completeness ─────────────────────────────────────────

class TestEvalResults:
    def test_zero_missing_tasks(self, seg: pd.DataFrame, eval_results: dict):
        for cp in CHECKPOINTS:
            results = eval_results[cp]
            missing = []
            for _, row in seg.iterrows():
                lm_eval = row["lm_eval_task"]
                if lm_eval not in results:
                    missing.append(row["task"])
            assert not missing, (
                f"[{cp}] Missing {len(missing)} tasks: {missing}"
            )

    def test_expected_metric_available(self, seg: pd.DataFrame, eval_results: dict):
        for cp in CHECKPOINTS:
            results = eval_results[cp]
            for _, row in seg.iterrows():
                lm_eval = row["lm_eval_task"]
                expected_metric = row["metric"]
                if lm_eval not in results:
                    continue
                task_data = results[lm_eval]
                found = any(
                    k.replace(",none", "").split(",")[0].strip() == expected_metric
                    for k in task_data.keys()
                )
                assert found, (
                    f"[{cp}] {row['task']}: metric '{expected_metric}' not in "
                    f"{list(task_data.keys())}"
                )


# ── README numbers verification ───────────────────────────────────────

class TestReadmeNumbers:
    """Verify that key numbers in the README match the raw eval data."""

    def _compute_scores(self, seg, eval_results):
        """Compute per-checkpoint scores from raw data."""
        scores = {}
        for cp in CHECKPOINTS:
            results = eval_results[cp]
            task_scores = []
            for _, row in seg.iterrows():
                lm_eval = row["lm_eval_task"]
                expected_metric = row["metric"]
                if lm_eval not in results:
                    continue
                task_data = results[lm_eval]
                for k, v in task_data.items():
                    clean = k.replace(",none", "").split(",")[0].strip()
                    if clean == expected_metric:
                        task_scores.append({
                            "task": row["task"],
                            "native_translated": row["native_translated"],
                            "categoria_paper": row["categoria_paper"],
                            "score": float(v),
                        })
                        break
            scores[cp] = pd.DataFrame(task_scores)
        return scores

    def test_all_scores(self, seg, eval_results):
        scores = self._compute_scores(seg, eval_results)
        expected_all = {
            "Qwen3-1.7B-Base": 0.6655,
            "TuQwen3-Base-LR1e5-run1": 0.6609,
            "QwenRolina3-Base": 0.6580,
        }
        for cp, expected in expected_all.items():
            actual = round(scores[cp]["score"].mean(), 4)
            assert actual == expected, (
                f"[{cp}] All score: expected {expected}, got {actual}"
            )

    def test_native_scores(self, seg, eval_results):
        scores = self._compute_scores(seg, eval_results)
        expected_native = {
            "Qwen3-1.7B-Base": 0.6862,
            "TuQwen3-Base-LR1e5-run1": 0.6902,
            "QwenRolina3-Base": 0.6891,
        }
        for cp, expected in expected_native.items():
            df = scores[cp]
            actual = round(df[df["native_translated"] == "Native"]["score"].mean(), 4)
            assert actual == expected, (
                f"[{cp}] Native score: expected {expected}, got {actual}"
            )

    def test_translated_scores(self, seg, eval_results):
        scores = self._compute_scores(seg, eval_results)
        expected_translated = {
            "Qwen3-1.7B-Base": 0.6544,
            "TuQwen3-Base-LR1e5-run1": 0.6451,
            "QwenRolina3-Base": 0.6412,
        }
        for cp, expected in expected_translated.items():
            df = scores[cp]
            actual = round(df[df["native_translated"] == "Translated"]["score"].mean(), 4)
            assert actual == expected, (
                f"[{cp}] Translated score: expected {expected}, got {actual}"
            )
