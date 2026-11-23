# Parameter-Efficient Fine-Tuning: LoRA and QLoRA

> Applies to both Classification Fine-Tuning and Instruction Fine-Tuning (*Build a Large Language Model from Scratch*, Appendix E).

**The Problem:** Full fine-tuning updates all billions of parameters in a pretrained model — which is expensive in GPU memory (weights + gradients + Adam optimizer states) and requires storing a full multi-gigabyte copy of the model for every downstream task.

---

## 1. LoRA (Low-Rank Adaptation)

Instead of updating the full pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, **LoRA freezes $W_0$** and injects a trainable **low-rank decomposition** $\Delta W = A \cdot B$ alongside it:

$$W' = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B \cdot A$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, rank $r \ll \min(d, k)$, and $\alpha$ is a scaling hyperparameter.

```
  Input x
     │
     ├──────────────────────┐
     ▼                      ▼
  ┌──────────────┐    ┌───────────┐
  │  Frozen W₀   │    │  Matrix A │  (d × r)  ← initialized with random Gaussian
  │  (d × k)     │    └─────┬─────┘
  └──────┬───────┘          │  low-rank bottleneck (rank r ≪ d)
         │            ┌─────▼─────┐
         │            │  Matrix B │  (r × k)  ← initialized to ALL ZEROS (ΔW = 0 at start)
         │            └─────┬─────┘
         ▼                  ▼
        ( + ) ◄──── scaled by α / r
         │
     Output h = x·W₀ + (α/r)·x·A·B
```

### Concrete Parameter Savings
```
  Suppose d = 4096, k = 4096, and rank r = 16:

  Full fine-tuning:   4096 × 4096           = 16,777,216 parameters per matrix
  LoRA (A + B):       (4096 × 16) + (16 × 4096) =    131,072 parameters  ← 128× fewer!
```

### Key Properties of LoRA
- **$W_0$ is frozen** (`requires_grad = False`) — the original pretrained weights never change and require no optimizer state memory.
- **Only $A$ and $B$ are trained** (typically $<1\%$ of total model parameters).
- **Zero-disruption initialization:** $A$ is initialized with small random Gaussian values, and $B$ is initialized to **zero** $\implies \Delta W = B \cdot A = 0$ at step 0, so the model starts at the exact pretrained behavior.
- **Zero inference latency:** After training, you can mathematically merge the weights once: $W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A$, adding **zero runtime overhead** at deployment.
- **Where is LoRA applied?** Typically to the attention projection matrices ($W_Q, W_K, W_V, W_O$) and optionally the FFN linear layers.

---

## 2. QLoRA (Quantized LoRA)

**QLoRA** pushes memory savings even further by quantizing the frozen base model weights $W_0$ to **4-bit precision**, while keeping the tiny LoRA adapters ($A, B$) in 16-bit (`BF16`):

| Component | QLoRA Setting |
|---|---|
| **Base model weights ($W_0$)** | 4-bit NormalFloat (**NF4**) — quantized and frozen |
| **LoRA adapters ($A$, $B$)** | **BF16** — trained with standard backpropagation |
| **Computation dtype** | Dequantizes $W_0$ to BF16 on-the-fly during forward/backward pass |
| **Optimizer states** | Paged AdamW (offloaded to CPU RAM when GPU VRAM spikes) |

**Why NormalFloat 4 (NF4)?** Pretrained neural network weights follow a zero-centered normal (Gaussian) distribution. NF4 places its $2^4 = 16$ quantization buckets at equal probability mass intervals of a standard normal distribution, minimizing quantization error. QLoRA enables fine-tuning a 70B parameter model on a single 48GB GPU.

