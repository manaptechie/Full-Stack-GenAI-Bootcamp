"""
Simple inference script for the IT Helpdesk AI Assistant.

Loads the final DPO-aligned LoRA adapter (produced by
`notebooks/dpo_alignment.ipynb`) on top of the Unsloth base model and lets a
user ask a question and get an answer.

Requirements:
    Must be run in an environment with a CUDA GPU and the same dependencies
    used in the training notebooks (unsloth, torch, transformers, peft).
    This script will not run on a CPU-only machine.

Usage:
    python src/inference.py
    python src/inference.py --question "How can I apply for reimbursement?"
"""

import argparse

BASE_MODEL = "unsloth/tinyllama-bnb-4bit"
DPO_ADAPTER_DIR = "Assignment-04/models/dpo_adapter"
MAX_SEQ_LENGTH = 1024

PROMPT_TEMPLATE = """Below is an IT Helpdesk support request. Write a helpful, professional response.

### Request:
{prompt}

### Response:
"""


def load_model(adapter_dir: str = DPO_ADAPTER_DIR):
    """Load the base model with the DPO-aligned LoRA adapter applied."""
    from unsloth import FastLanguageModel
    import os

    model_name = adapter_dir if os.path.isdir(adapter_dir) else BASE_MODEL
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_answer(question: str, model=None, tokenizer=None) -> str:
    """Generate an answer for a single IT Helpdesk question."""
    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    prompt = PROMPT_TEMPLATE.format(prompt=question)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200, use_cache=True)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()


def main():
    parser = argparse.ArgumentParser(description="IT Helpdesk AI Assistant inference")
    parser.add_argument(
        "--question",
        type=str,
        default="How can I apply for reimbursement?",
        help="The IT Helpdesk question to ask the model",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default=DPO_ADAPTER_DIR,
        help="Path to the DPO-aligned LoRA adapter directory",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter_dir)
    answer = generate_answer(args.question, model, tokenizer)

    print(f"Question: {args.question}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
