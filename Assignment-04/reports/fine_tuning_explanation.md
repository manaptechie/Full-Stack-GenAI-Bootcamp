# Fine-Tuning Concepts Explained

## Why full fine-tuning is expensive

Full fine-tuning updates every parameter of the model. For even a small 1B–7B
parameter model this means storing and updating billions of gradients and
optimizer states in GPU memory (often 3–4x the model size for Adam optimizer
states), which quickly exceeds consumer/free-tier GPU memory and takes far
longer to train than parameter-efficient approaches.

## What LoRA does

LoRA (Low-Rank Adaptation) freezes the original model weights and injects a
small pair of trainable low-rank matrices (rank `r`) into selected layers
(typically the attention projection and MLP layers). Only these small matrices
are trained, drastically reducing the number of trainable parameters (often
<1% of the full model) while still allowing the model to adapt to a new
domain or task.

## What QLoRA does

QLoRA combines LoRA with 4-bit quantization of the frozen base model weights
(using NF4 quantization) plus double quantization and paged optimizers. The
base model is loaded in 4-bit precision to save memory, while the LoRA
adapters are still trained in higher precision (e.g. bf16/fp16), giving
near full-fine-tuning quality at a fraction of the memory cost.

## Why QLoRA is useful on limited GPU

Because the frozen base weights only need ~4 bits per parameter instead of 16
or 32, QLoRA makes it possible to fine-tune models that would otherwise not
fit in a free-tier GPU (e.g. a 16GB T4 on Google Colab), while training only
a small number of LoRA parameters on top.

## What is non-instruction fine-tuning?

Non-instruction fine-tuning (a.k.a. continued pre-training / domain adaptation)
continues training the base model on raw, unstructured domain text using the
standard causal language modeling objective (predict the next token). There
is no instruction/response structure — the goal is simply to shift the
model's language distribution toward domain-specific vocabulary, tone, and
style before teaching it to follow instructions.

## What is instruction fine-tuning?

Instruction fine-tuning (SFT, supervised fine-tuning) trains the model on
paired `(instruction, response)` examples so that, given a user request, it
learns to produce a helpful, on-format response instead of just continuing
the text. This is what turns a raw language model into an assistant that can
answer domain-specific questions directly.

## What is DPO?

Direct Preference Optimization (DPO) is a preference-alignment technique that
trains the model directly on `(prompt, chosen, rejected)` triples, increasing
the likelihood of the chosen response relative to the rejected response
(relative to a frozen reference model), without needing a separate reward
model or reinforcement learning loop (unlike classic RLHF/PPO).

## Difference between SFT and DPO

- **SFT** teaches the model *what a good answer looks like* by training on
  single "correct" responses (behavior cloning).
- **DPO** teaches the model *which of two answers is better* by training on
  contrastive (chosen vs rejected) pairs, refining response quality, safety,
  and style beyond what SFT alone can achieve.

## Hyperparameters Used

| Stage | Rank (r) | Alpha | Dropout | Learning Rate | Batch Size (effective) |
|---|---|---|---|---|---|
| Stage 1: Non-instruction FT | 16 | 16 | 0.05 | 2e-4 | 2 x 4 grad-accum = 8 |
| Stage 2: Instruction FT (SFT) | 16 | 16 | 0.05 | 2e-4 | 2 x 4 grad-accum = 8 |
| Stage 3: DPO alignment | 16 | 16 | 0.05 | 5e-5 | 1 x 4 grad-accum = 4 |

_Update these values with the actual settings used once the notebooks are run on a GPU._
