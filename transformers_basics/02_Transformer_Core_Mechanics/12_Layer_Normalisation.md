## 9. Layer Normalisation

Layer Norm rescales each token's representation to have zero mean and unit variance, then applies learned scale $\delta$ and shift $\beta$:

$$LN(x) = \delta \cdot \hat{x} + \beta \qquad \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

$$\mu = \frac{1}{d}\sum_{i=1}^{d} x_i \qquad \sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2$$

- $\delta, \beta$ — learned parameters (scale and shift)
- $\epsilon$ — small constant (e.g. $10^{-5}$) to prevent division by zero

> **Biased vs unbiased variance:** The formula divides by $d$ (biased). PyTorch's `torch.var()` defaults to Bessel's correction ($d-1$), but `nn.LayerNorm` correctly uses biased variance to match the paper. At $d_{model}=768$ the difference is $768/767 \approx 0.13\%$ — negligible in practice.

### Worked example — token [1.0, 2.0, 3.0, 4.0]

$$\mu = 2.5 \qquad \sigma^2 = 1.25 \qquad \sqrt{\sigma^2} \approx 1.118$$

$$\hat{x} = \left[\frac{-1.5}{1.118},\ \frac{-0.5}{1.118},\ \frac{0.5}{1.118},\ \frac{1.5}{1.118}\right] = [-1.34,\ -0.45,\ 0.45,\ 1.34]$$

> **Key property:** Layer Norm operates on **each token row independently** — it never looks at other tokens in the sequence. This makes it stable for variable-length inputs, unlike Batch Normalisation.

### What is a SubLayer?

**SubLayer** = one of the two main operations inside a transformer block — either Multi-Head Attention or FFN. Each block has two sublayers, each wrapped with residual + norm:

```
  Transformer block:
  ┌──────────────────────────────────────────┐
  │  Sublayer 1: Multi-Head (Self-)Attention │
  │     wrapped with: residual + LayerNorm   │
  │                                          │
  │  Sublayer 2: Feed-Forward Network        │
  │     wrapped with: residual + LayerNorm   │
  └──────────────────────────────────────────┘
```

### Pre-Norm vs Post-Norm

| Variant | Formula | Era |
|---|---|---|
| **Post-Norm** | $\text{Output} = \text{LayerNorm}(x + \text{SubLayer}(x))$ | Original 2017 paper |
| **Pre-Norm** | $\text{Output} = x + \text{SubLayer}(\text{LayerNorm}(x))$ | Modern standard |

```
  Post-Norm:                         Pre-Norm:
  x ─────────────────────────┐       x ──► LayerNorm ──► SubLayer ──┐
  x ──► SubLayer ──► Add ──► LN      x ─────────────────────── Add ──► output
                                          ↑
                                    residual carries clean unnormalised x
```

**Why Pre-Norm won:** In Post-Norm, the residual is added before normalising — at the start of training this sum can be large and noisy. In Pre-Norm, the unnormalised $x$ flows through the residual path always, keeping gradients well-behaved without warmup tricks. Essential at GPT-3+ scale.

### Who uses what

| Model | Norm type | Placement |
|---|---|---|
| Original Transformer (2017) | Layer Norm | Post-Norm |
| BERT | Layer Norm | Post-Norm |
| GPT-2 / GPT-3 | Layer Norm | Pre-Norm |
| LLaMA 1/2/3, Mistral, Gemma, T5 | **RMS Norm** | Pre-Norm |

### RMS Norm

$$\text{RMSNorm}(x) = \delta \cdot \frac{x}{\text{RMS}(x)} \qquad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}$$

Drops the mean-subtraction step entirely — faster to compute, no measurable quality loss. Modern default alongside Pre-Norm.

---
