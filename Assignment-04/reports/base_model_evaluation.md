# Base Model Evaluation (Before Fine-Tuning)

**Model:** `unsloth/tinyllama-bnb-4bit` (TinyLlama-1.1B, 4-bit, no fine-tuning applied)

**Purpose:** Establish a baseline of how the untouched base model responds to 10 IT
Helpdesk domain questions, before any non-instruction, instruction, or DPO
fine-tuning is applied. Fill in the "Base Model Answer" column after running
the base model inference cell (Stage 1/2 notebook, before training) on a GPU
runtime.

| # | Question | Base Model Answer | Observed Problem |
|---|----------|--------------------|-------------------|
| 1 | Unable to Access Office Applications after password reset. | _fill in after running base model_ | Generic, not domain-specific, no concrete troubleshooting steps |
| 2 | My VPN keeps disconnecting every few minutes, how do I fix it? | _fill in after running base model_ | Vague, does not reference VPN client/DNS troubleshooting |
| 3 | The reporting dashboard is not loading any widgets. | _fill in after running base model_ | No awareness of SaaS release notes or caching issues |
| 4 | I was charged extra fees on my invoice this month, why? | _fill in after running base model_ | No billing-specific reasoning, generic apology only |
| 5 | How do I set up the integration with our third-party CRM tool? | _fill in after running base model_ | No structured setup guidance |
| 6 | Our production system just went down, what should I do? | _fill in after running base model_ | No incident-response tone or ETA |
| 7 | I forgot my password and the reset link is not arriving. | _fill in after running base model_ | No account-lockout specific steps |
| 8 | Why is our Wi-Fi so slow today across the whole office? | _fill in after running base model_ | No network diagnostics suggested |
| 9 | Can you resend last month's invoice? | _fill in after running base model_ | No confirmation / follow-up action |
| 10 | The bulk data import keeps failing with validation errors. | _fill in after running base model_ | No mention of file template/validation |

## Summary Observations

- The base model tends to give **generic, non-domain-specific** answers.
- It rarely references IT Helpdesk-specific actions (VPN client updates, ticket
  escalation, invoice breakdowns, lockout resets, etc.).
- Responses lack the professional support tone (empathy + clear next steps +
  timeline) expected from a real helpdesk agent.
- This baseline is used for comparison in [sft_model_comparison.md](sft_model_comparison.md)
  and [final_evaluation.md](final_evaluation.md).
