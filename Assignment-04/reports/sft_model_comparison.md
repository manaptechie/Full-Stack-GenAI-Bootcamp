# Base Model vs Instruction Fine-Tuned (SFT) Model Comparison

**Base model:** `unsloth/tinyllama-bnb-4bit` (no fine-tuning)
**SFT model:** Base model + Stage 1 non-instruction adapter + Stage 2 instruction fine-tuning (`models/sft_adapter`)

Run the same 10 questions through both models (base model answers copied from
[base_model_evaluation.md](base_model_evaluation.md), SFT answers from the
`instruction_finetuning.ipynb` inference cell) and fill in the table below.

| # | Question | Base Model Answer | SFT Model Answer | Improvement Notes |
|---|----------|--------------------|--------------------|--------------------|
| 1 | Unable to Access Office Applications after password reset. | _see base report_ | _fill in after SFT run_ | |
| 2 | My VPN keeps disconnecting every few minutes, how do I fix it? | _see base report_ | _fill in after SFT run_ | |
| 3 | The reporting dashboard is not loading any widgets. | _see base report_ | _fill in after SFT run_ | |
| 4 | I was charged extra fees on my invoice this month, why? | _see base report_ | _fill in after SFT run_ | |
| 5 | How do I set up the integration with our third-party CRM tool? | _see base report_ | _fill in after SFT run_ | |
| 6 | Our production system just went down, what should I do? | _see base report_ | _fill in after SFT run_ | |
| 7 | I forgot my password and the reset link is not arriving. | _see base report_ | _fill in after SFT run_ | |
| 8 | Why is our Wi-Fi so slow today across the whole office? | _see base report_ | _fill in after SFT run_ | |
| 9 | Can you resend last month's invoice? | _see base report_ | _fill in after SFT run_ | |
| 10 | The bulk data import keeps failing with validation errors. | _see base report_ | _fill in after SFT run_ | |

## Evaluation Criteria

Rate each answer 1-5 on the following criteria and summarize below:

| Criteria | Base Model (avg) | SFT Model (avg) |
|---|---|---|
| Correctness | | |
| Domain accuracy | | |
| Clarity | | |
| Safety | | |
| Helpfulness | | |
| Less generic response | | |
| Better domain-specific behavior | | |

## Summary Observations

- _Fill in after running both models: describe how much more domain-specific,_
  _actionable, and professionally toned the SFT answers are compared to the base model._
