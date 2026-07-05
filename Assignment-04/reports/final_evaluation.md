# Final Evaluation: Base vs Instruction Fine-Tuned (SFT) vs DPO-Aligned Model

This report compares all three stages of the IT Helpdesk AI Assistant pipeline:

1. **Base model** — `unsloth/tinyllama-bnb-4bit`, no fine-tuning
2. **SFT model** — Stage 1 (non-instruction) + Stage 2 (instruction fine-tuning)
3. **DPO model** — Stage 3, preference-aligned on top of the SFT model (`models/dpo_adapter`)

| # | Question | Base Model Answer | SFT Model Answer | DPO Model Answer |
|---|----------|--------------------|--------------------|--------------------|
| 1 | Unable to Access Office Applications after password reset. | _see base report_ | _see sft report_ | _fill in after DPO run_ |
| 2 | My VPN keeps disconnecting every few minutes, how do I fix it? | _see base report_ | _see sft report_ | _fill in after DPO run_ |
| 3 | The reporting dashboard is not loading any widgets. | _see base report_ | _see sft report_ | _fill in after DPO run_ |
| 4 | I was charged extra fees on my invoice this month, why? | _see base report_ | _see sft report_ | _fill in after DPO run_ |
| 5 | How do I set up the integration with our third-party CRM tool? | _see base report_ | _see sft report_ | _fill in after DPO run_ |
| 6 | Our production system just went down, what should I do? | _see base report_ | _see sft report_ | _fill in after DPO run_ |
| 7 | I forgot my password and the reset link is not arriving. | _see base report_ | _see sft report_ | _fill in after DPO run_ |
| 8 | Why is our Wi-Fi so slow today across the whole office? | _see base report_ | _see sft report_ | _fill in after DPO run_ |
| 9 | Can you resend last month's invoice? | _see base report_ | _see sft report_ | _fill in after DPO run_ |
| 10 | The bulk data import keeps failing with validation errors. | _see base report_ | _see sft report_ | _fill in after DPO run_ |

## Evaluation Criteria (1-5 scale, averaged across all 10 questions)

| Criteria | Base Model | SFT Model | DPO Model |
|---|---|---|---|
| Correctness | | | |
| Helpfulness | | | |
| Domain accuracy | | | |
| Safety | | | |
| Tone | | | |
| Clarity | | | |
| Hallucination reduction | | | |
| Professional response quality | | | |

## Final Observations

- _Fill in after all three stages are trained and evaluated._
- Describe the incremental improvement from base -> SFT -> DPO.
- Note any regressions or trade-offs observed (e.g. verbosity, repetition, latency).

## Challenges Faced

- _e.g. no local GPU, dataset scale limitations, training instability, etc._

## Future Improvements

- _e.g. larger/more diverse dataset, longer training, larger base model, human-labeled preference pairs, RLHF/ORPO experiments._
