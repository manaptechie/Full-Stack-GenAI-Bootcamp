"""
Auto-fill the evaluation report tables in `reports/` using the JSON answer
files produced by the notebooks' evaluation cells:

    reports/base_model_answers.json   <- notebooks/non_instruction_finetuning.ipynb (Stage 0 baseline)
    reports/sft_model_answers.json    <- notebooks/instruction_finetuning.ipynb (Stage 2)
    reports/dpo_model_answers.json    <- notebooks/dpo_alignment.ipynb (Stage 3)

Each JSON file is a list of {"question": ..., "answer": ...} objects in the
same order as `src/eval_questions.EVAL_QUESTIONS`.

This script is pure Python (no torch/unsloth/datasets dependency) and is
meant to be run locally *after* copying the JSON files back from the
Colab/Kaggle GPU runtime where the notebooks were executed.

Usage:
    cd Assignment-04
    python src/fill_reports.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from eval_questions import EVAL_QUESTIONS

REPORTS_DIR = Path(__file__).parent.parent / "reports"

ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|")


def load_answers(name: str):
    """Load an answers JSON file as a list indexed the same as EVAL_QUESTIONS, or None if missing."""
    path = REPORTS_DIR / name
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    answers = [None] * len(EVAL_QUESTIONS)
    for record in records:
        if record["question"] in EVAL_QUESTIONS:
            answers[EVAL_QUESTIONS.index(record["question"])] = record["answer"]
    return answers


def to_cell(text: str) -> str:
    """Escape an answer so it fits safely as a single markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def fill_table(path: Path, placeholder_answers: dict):
    """
    Replace known placeholder cells in a markdown table with real answers.

    placeholder_answers: {placeholder_substring: answers_list_or_None}
    Rows are matched by their leading "| N |" row number (1-indexed),
    corresponding to EVAL_QUESTIONS[N - 1].
    """
    if not path.exists():
        print(f"Skipped (not found): {path}")
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    changed = False

    for i, line in enumerate(lines):
        match = ROW_RE.match(line)
        if not match:
            continue
        row_num = int(match.group(1))
        idx = row_num - 1
        if idx < 0 or idx >= len(EVAL_QUESTIONS):
            continue

        new_line = line
        for placeholder, answers in placeholder_answers.items():
            if answers is None or answers[idx] is None:
                continue
            new_line = new_line.replace(placeholder, to_cell(answers[idx]), 1)
        if new_line != line:
            lines[i] = new_line
            changed = True

    if changed:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Updated: {path}")
    else:
        print(f"No matching answers found for: {path} (run the notebooks and copy the JSON files first)")


def main():
    base_answers = load_answers("base_model_answers.json")
    sft_answers = load_answers("sft_model_answers.json")
    dpo_answers = load_answers("dpo_model_answers.json")

    fill_table(
        REPORTS_DIR / "base_model_evaluation.md",
        {"_fill in after running base model_": base_answers},
    )
    fill_table(
        REPORTS_DIR / "sft_model_comparison.md",
        {
            "_see base report_": base_answers,
            "_fill in after SFT run_": sft_answers,
        },
    )
    fill_table(
        REPORTS_DIR / "final_evaluation.md",
        {
            "_see base report_": base_answers,
            "_see sft report_": sft_answers,
            "_fill in after DPO run_": dpo_answers,
        },
    )


if __name__ == "__main__":
    main()
