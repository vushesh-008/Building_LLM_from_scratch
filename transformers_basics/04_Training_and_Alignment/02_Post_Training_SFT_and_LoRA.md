# Post-Training: SFT and LoRA

Pre-training teaches a model to predict the next token. That alone produces a model that continues text — not one that follows instructions, reasons, or refuses harmful requests. Post-training is the pipeline that transforms a raw pretrained base into a usable assistant.

```
  Base model (pretrained)
       │
  SFT  ── teaches instruction following
       │
  RLHF / DPO / GRPO  ── aligns with human preferences, improves reasoning
       │
  Deployed assistant
```

## Supervised Fine-Tuning (SFT)

The simplest step: fine-tune on a dataset of (prompt, response) pairs where responses are human-written or filtered to be high quality.

```
  Training example:
  ┌──────────────────────────────────────────────────────┐
  │ Prompt:   "Explain gradient descent in simple terms" │
  │ Response: "Imagine you're lost in hilly terrain..."  │
  └──────────────────────────────────────────────────────┘

  Token sequence fed to the model:
  [system] [prompt tokens] [response tokens]

  Loss mask (same label=-100 trick as VLM):
  [  -100 ] [    -100    ] [ CE loss here  ]
                            ↑ only response tokens supervised
```

**Chat templates** define how the system/user/assistant turns are formatted. Each model family has its own format:

```
  LLaMA 3 chat template:
  <|begin_of_text|>
  <|start_header_id|>system<|end_header_id|>
  You are a helpful assistant.<|eot_id|>
  <|start_header_id|>user<|end_header_id|>
  Explain gradient descent.<|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>
  Imagine you're lost...
```

The model learns to predict only the assistant turns — the prompt and system message are context, not targets.

---

## LoRA and QLoRA

**The problem:** full fine-tuning updates all billions of parameters — expensive in GPU memory and compute, and risks catastrophic forgetting.

**LoRA (Low-Rank Adaptation):** instead of updating the full weight matrix $W \in \mathbb{R}^{d \times k}$, add a low-rank correction:

$$W' = W_0 + \Delta W = W_0 + B \cdot A$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and rank $r \ll d$.

```
  d = 4096, k = 4096, r = 16:

  Full fine-tuning:   4096 × 4096 = 16.7M parameters per matrix
  LoRA:               4096×16 + 16×4096 = 131K parameters  ← 128× fewer
```

- $W_0$ is **frozen** — the original pretrained weights never change
- Only $A$ and $B$ are trained (typically $<1\%$ of total parameters)
- $A$ initialised randomly, $B$ initialised to zero → $\Delta W = 0$ at start (no disruption)
- At inference: $W' = W_0 + BA$ is merged — zero runtime overhead

LoRA is typically applied to the Q, K, V, and output projection matrices in attention.

**QLoRA:** adds 4-bit quantisation of the frozen base model weights:

| Component | QLoRA setting |
|---|---|
| Base model weights ($W_0$) | 4-bit NormalFloat (NF4) — quantised, frozen |
| LoRA adapters ($A$, $B$) | BF16 — trained normally |
| Optimizer states | Paged (offloaded to CPU RAM when GPU is tight) |

NF4 is designed for normally-distributed weights (which pretrained LLM weights are) — it places quantisation levels at equal probability mass intervals, minimising error. QLoRA enables fine-tuning a 70B model on a single 48GB GPU.
