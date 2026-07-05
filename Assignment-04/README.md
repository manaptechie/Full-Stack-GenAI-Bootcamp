# IT Helpdesk AI Assistant — Domain-Specific Fine-Tuning with Unsloth

## Project Title

**Building a Domain-Specific AI Assistant using Unsloth: IT Helpdesk Assistant**

## Domain Selected

**IT Helpdesk Assistant** — an internal support assistant that answers
questions related to:

- System Interruptions
- Connectivity Issues
- Issues with SaaS Platform Functionality
- Application access related issues
- Product assistance requests
- Invoice inquiries

## Business Problem

As a GenAI Engineer, the goal is to build an internal AI assistant for the
IT Helpdesk domain that understands domain-specific terminology, answers
employee/customer questions clearly, and gives more accurate, professional,
and actionable responses than a generic base model.

## Pipeline Overview

```
Base Model (TinyLlama-1.1B, Unsloth, 4-bit)
   ↓
Stage 1: Non-Instruction Fine-Tuning   (raw domain text, causal LM)
   ↓
Stage 2: Instruction Fine-Tuning (SFT) (instruction/response pairs)
   ↓
Stage 3: DPO Preference Alignment      (chosen vs rejected pairs)
   ↓
Final Domain-Specific IT Helpdesk AI Assistant
```

## Dataset Details

All three datasets were built from **AI-generated, template-based domain
content** covering the six IT Helpdesk categories above (see
`data/generate_domain_data.py`), then cleaned and verified against the
assignment's minimum requirements. This approach was chosen instead of the
public `Tobi-Bueck/customer-support-tickets` Hugging Face dataset because it
avoids external downloads/auth requirements and keeps the pipeline fully
reproducible offline; the public dataset script (`data/prepare_dataset.py`)
is still included as an alternative data source.

| File | Rows | Format |
|---|---|---|
| `data/non_instruction_data.txt` | 50 paragraphs | Raw domain text (blank-line separated) |
| `data/instruction_dataset.jsonl` | 120 examples | `{"instruction": ..., "response": ...}` |
| `data/preference_dataset.jsonl` | 120 examples | `{"prompt": ..., "chosen": ..., "rejected": ...}` |

To regenerate the datasets:

```bash
cd Assignment-04/data
python generate_domain_data.py
```

## Base Model Used

`unsloth/tinyllama-bnb-4bit` (TinyLlama-1.1B, 4-bit quantized) — chosen from
the assignment's recommended small-model list for fast iteration on a free
GPU tier (Colab T4 / Kaggle GPU).

> **Important — GPU requirement:** Unsloth requires a CUDA GPU. This
> repository was authored on a CPU-only machine, so the three notebooks are
> designed to be run on **Google Colab or Kaggle with a GPU runtime**. The
> datasets, reports templates, and inference script are fully prepared here;
> run the notebooks in order on a GPU to produce the actual trained adapters
> and fill in the evaluation tables.

## Non-Instruction Fine-Tuning Approach

`notebooks/non_instruction_finetuning.ipynb`:
1. Load and clean `data/non_instruction_data.txt`.
2. Chunk paragraphs into ~512-character blocks.
3. Load the base model with `FastLanguageModel.from_pretrained`.
4. Apply LoRA and train with `SFTTrainer` using plain causal-LM packing
   (no prompt/response structure) to adapt the model to domain language.
5. Save the adapter to `models/non_instruction_adapter`.

## Instruction Fine-Tuning Approach

`notebooks/instruction_finetuning.ipynb`:
1. Continue from the Stage 1 adapter (or base model if unavailable).
2. Format `data/instruction_dataset.jsonl` into a `### Request / ### Response`
   prompt template.
3. Apply LoRA and train with `SFTTrainer`.
4. Save the adapter to `models/sft_adapter`.
5. Run inference on 10 fixed evaluation questions for comparison.

## DPO Alignment Approach

`notebooks/dpo_alignment.ipynb`:
1. Load the Stage 2 SFT adapter.
2. Format `data/preference_dataset.jsonl` into `prompt` / `chosen` / `rejected`.
3. Train with `trl.DPOTrainer` (`beta=0.1`), using the base model as the
   implicit reference.
4. Save the final adapter to `models/dpo_adapter`.
5. Re-run the same 10 evaluation questions for the final comparison.

## LoRA / QLoRA Configuration

| Stage | Rank (r) | Alpha | Dropout | Learning Rate | Effective Batch Size |
|---|---|---|---|---|---|
| Non-instruction FT | 16 | 16 | 0.05 | 2e-4 | 8 (2 × 4 grad-accum) |
| Instruction FT (SFT) | 16 | 16 | 0.05 | 2e-4 | 8 (2 × 4 grad-accum) |
| DPO alignment | 16 | 16 | 0.05 | 5e-5 | 4 (1 × 4 grad-accum) |

Quantization: 4-bit (QLoRA-style) via `load_in_4bit=True`.

## Training Screenshots / Logs

_Add screenshots or trainer logs here after running the notebooks on a GPU
(e.g. Colab training loss curves, `trainer_stats` output)._

## Before vs After Output Comparison

See:
- [`reports/base_model_evaluation.md`](Assignment-04/reports/base_model_evaluation.md) — base model on 10 questions
- [`reports/sft_model_comparison.md`](Assignment-04/reports/sft_model_comparison.md) — base vs SFT
- [`reports/final_evaluation.md`](Assignment-04/reports/final_evaluation.md) — base vs SFT vs DPO
- [`reports/fine_tuning_explanation.md`](Assignment-04/reports/fine_tuning_explanation.md) — LoRA/QLoRA/SFT/DPO concepts explained

## Final Observations

_Fill in after running all three notebooks on a GPU: summarize how much the
model's domain accuracy, tone, and helpfulness improved from base → SFT → DPO._

## Challenges Faced

- No local GPU available for training; notebooks must be run on Colab/Kaggle.
- Public HF dataset (`Tobi-Bueck/customer-support-tickets`) requires network
  access and a working `datasets`/`torch` install, so a template-based
  synthetic dataset generator was used instead to keep the pipeline
  self-contained and reproducible.

## Future Improvements

- Replace template-generated data with real, larger-scale IT helpdesk tickets
  (e.g. the public `Tobi-Bueck/customer-support-tickets` dataset via
  `data/prepare_dataset.py`) once running in a GPU environment with internet access.
- Experiment with larger base models (Llama-3.2-1B, Gemma 2B) and ORPO as an
  alternative to DPO.
- Add automated LLM-as-judge scoring for the evaluation reports instead of
  manual scoring.

## Repository Structure

```
Assignment-04/
├── data/
│   ├── generate_domain_data.py       # synthetic dataset generator (used)
│   ├── prepare_dataset.py            # public HF dataset alternative
│   ├── non_instruction_data.txt
│   ├── instruction_dataset.jsonl
│   └── preference_dataset.jsonl
├── notebooks/
│   ├── non_instruction_finetuning.ipynb
│   ├── instruction_finetuning.ipynb
│   └── dpo_alignment.ipynb
├── reports/
│   ├── base_model_evaluation.md
│   ├── sft_model_comparison.md
│   ├── final_evaluation.md
│   └── fine_tuning_explanation.md
├── src/
│   └── inference.py
└── Fine-Tuning Project in Python.docx   # original assignment brief
```

## Public Dataset Reference

IT Helpdesk Assistant dataset (alternative data source):
https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets
