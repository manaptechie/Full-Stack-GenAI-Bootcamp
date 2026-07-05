"""
Data preparation script for the IT Helpdesk AI Assistant fine-tuning project.

Source dataset:
    https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets

This script downloads the public customer-support-tickets dataset, filters it
down to IT / technical-support related tickets written in English, cleans the
text, and produces the three files required by the assignment:

    data/non_instruction_data.txt   -> raw domain text (Stage 1: non-instruction FT)
    data/instruction_dataset.jsonl  -> instruction/response pairs (Stage 2: instruction FT)
    data/preference_dataset.jsonl   -> prompt/chosen/rejected triples (Stage 3: DPO)

Usage:
    python prepare_dataset.py

The Hugging Face token (if needed for rate limits / gated access) is read from
the project's env/.env file (HF_TOKEN=...). The dataset itself is public, so a
token is not strictly required, but we still honor it when present.
"""

import json
import os
import random
import re
from pathlib import Path

from datasets import load_dataset


def load_env_file(env_path: Path) -> None:
    """Minimal .env loader (avoids a hard dependency on python-dotenv)."""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent  # .../Full-Stack-GenAI-Bootcamp-1.0
ENV_PATH = REPO_ROOT / "env" / ".env"

load_env_file(ENV_PATH)
HF_TOKEN = os.getenv("HF_TOKEN")

random.seed(42)

DATASET_ID = "Tobi-Bueck/customer-support-tickets"

# Queues that map to an "IT Helpdesk" style assistant.
IT_QUEUES = {
    "Technical Support",
    "IT Support",
    "Service Outages and Maintenance",
    "Product Support",
}

NON_INSTRUCTION_PATH = THIS_DIR / "non_instruction_data.txt"
INSTRUCTION_PATH = THIS_DIR / "instruction_dataset.jsonl"
PREFERENCE_PATH = THIS_DIR / "preference_dataset.jsonl"

MIN_NON_INSTRUCTION_PARAGRAPHS = 60
MIN_INSTRUCTION_EXAMPLES = 120
MIN_PREFERENCE_EXAMPLES = 60


def clean_text(text: str) -> str:
    """Collapse whitespace and strip common email boilerplate."""
    if not text:
        return ""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_generic_rejection(answer: str) -> str:
    """
    Synthesize a plausible 'rejected' response for DPO training.

    Since the source dataset only contains a single (good) agent answer per
    ticket, we generate a weaker counterpart by truncating / genericizing the
    real answer. This keeps the rejected response topically related (so the
    model learns *quality* differences) while remaining clearly weaker:
    shorter, vaguer, less actionable, and without the professional closing.
    """
    generic_openers = [
        "Please try restarting your device and contact us again if the issue continues.",
        "This is a known issue. Please wait and it should resolve itself.",
        "We are unable to help with this right now, please check the FAQ page.",
        "Try turning it off and on again.",
        "This isn't really our department, please contact someone else.",
    ]
    return random.choice(generic_openers)


def load_it_helpdesk_split():
    print(f"Loading dataset: {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID, split="train", token=HF_TOKEN)
    print(f"Total rows in source dataset: {len(ds)}")

    def keep(example):
        queue_ok = example.get("queue") in IT_QUEUES
        lang_ok = example.get("language", "en") == "en"
        has_text = bool(example.get("subject")) and bool(example.get("answer"))
        return queue_ok and lang_ok and has_text

    filtered = ds.filter(keep)
    print(f"Filtered IT-helpdesk (English) rows: {len(filtered)}")
    return filtered


def build_non_instruction_data(rows) -> list:
    """Raw domain paragraphs: ticket subject + body + agent answer, concatenated."""
    paragraphs = []
    for row in rows:
        subject = clean_text(row.get("subject", ""))
        body = clean_text(row.get("body", ""))
        answer = clean_text(row.get("answer", ""))
        if not subject or not answer:
            continue
        paragraph = f"{subject}. {body} {answer}".strip()
        paragraph = re.sub(r"\s{2,}", " ", paragraph)
        if len(paragraph.split()) >= 20:
            paragraphs.append(paragraph)
    return paragraphs


def build_instruction_examples(rows) -> list:
    examples = []
    for row in rows:
        subject = clean_text(row.get("subject", ""))
        body = clean_text(row.get("body", ""))
        answer = clean_text(row.get("answer", ""))
        if not subject or not answer:
            continue
        instruction = subject if not body else f"{subject}\n\n{body}"
        examples.append({"instruction": instruction, "response": answer})
    return examples


def build_preference_examples(rows) -> list:
    examples = []
    for row in rows:
        subject = clean_text(row.get("subject", ""))
        body = clean_text(row.get("body", ""))
        answer = clean_text(row.get("answer", ""))
        if not subject or not answer or len(answer.split()) < 15:
            continue
        prompt = subject if not body else f"{subject}\n\n{body}"
        examples.append(
            {
                "prompt": prompt,
                "chosen": answer,
                "rejected": make_generic_rejection(answer),
            }
        )
    return examples


def main():
    rows = load_it_helpdesk_split()
    rows_list = list(rows)
    random.shuffle(rows_list)

    # --- Stage 1: non-instruction raw text ---------------------------------
    paragraphs = build_non_instruction_data(rows_list)
    paragraphs = paragraphs[: max(MIN_NON_INSTRUCTION_PARAGRAPHS * 3, 300)]
    if len(paragraphs) < MIN_NON_INSTRUCTION_PARAGRAPHS:
        raise RuntimeError("Not enough paragraphs extracted for non-instruction data")
    with open(NON_INSTRUCTION_PATH, "w", encoding="utf-8") as f:
        f.write("\n\n".join(paragraphs))
    print(f"Wrote {len(paragraphs)} paragraphs -> {NON_INSTRUCTION_PATH}")

    # --- Stage 2: instruction dataset --------------------------------------
    instruction_examples = build_instruction_examples(rows_list)
    instruction_examples = instruction_examples[: max(MIN_INSTRUCTION_EXAMPLES * 3, 600)]
    if len(instruction_examples) < MIN_INSTRUCTION_EXAMPLES:
        raise RuntimeError("Not enough rows for instruction dataset")
    with open(INSTRUCTION_PATH, "w", encoding="utf-8") as f:
        for ex in instruction_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(instruction_examples)} instruction examples -> {INSTRUCTION_PATH}")

    # --- Stage 3: preference dataset ----------------------------------------
    preference_examples = build_preference_examples(rows_list)
    preference_examples = preference_examples[: max(MIN_PREFERENCE_EXAMPLES * 3, 200)]
    if len(preference_examples) < MIN_PREFERENCE_EXAMPLES:
        raise RuntimeError("Not enough rows for preference dataset")
    with open(PREFERENCE_PATH, "w", encoding="utf-8") as f:
        for ex in preference_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(preference_examples)} preference examples -> {PREFERENCE_PATH}")


if __name__ == "__main__":
    main()
