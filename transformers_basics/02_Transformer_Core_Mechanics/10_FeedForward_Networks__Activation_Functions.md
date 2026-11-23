## 9. Feed-Forward Networks & Activation Functions

Every transformer block applies a Feed-Forward Network (FFN) independently to each token position after attention:

$$\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$$

The FFN expands the representation to a wider hidden dimension (typically $4 \times d_{model}$) and projects back down. The activation function $\sigma$ is the only source of non-linearity in the block.

### ReLU — Original Transformer (2017)

$$\text{ReLU}(x) = \max(0, x)$$

```
        │        /
        │       /
   ─────┼─────/─────  x
        │
```

Simple and fast. Problem: **dying ReLU** — neurons whose pre-activation is negative output zero and receive zero gradient, so their weights stop updating. A large fraction of the network can permanently die during training.

### GELU — BERT, GPT-2, GPT-3

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the Gaussian CDF. Approximated as:

$$\text{GELU}(x) \approx 0.5x\!\left(1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}}\,(x + 0.044715x^3)\right)\right)$$

```
        │          /
        │         /
   ─────┼────────/──────  x
        │    ___/
        │   /       ← slightly negative for small negative x (smooth, not hard-zero)
```

GELU doesn't hard-zero negative inputs — it down-weights them smoothly. Better gradient flow than ReLU.

### SwiGLU — LLaMA, Mistral, Gemma, PaLM (modern default)

SwiGLU changes the FFN structure, not just the activation. It adds a **gating mechanism**:

$$\text{SwiGLU}(x) = \text{Swish}(W_1 x) \odot (W_3 x) \qquad \text{where} \quad \text{Swish}(x) = x \cdot \sigma(x)$$

```
  Standard FFN (GELU):
  x ──► Linear (W₁) ──► GELU ──► Linear (W₂) ──► output

  SwiGLU FFN:
  x ──► Linear (W₁) ──► Swish ──┐
                                  ├── ⊙ ──► Linear (W₂) ──► output
  x ──► Linear (W₃) ─────────────┘
              ↑
        element-wise multiply (the "gate")
```

- $W_1 x$ through Swish: activated features
- $W_3 x$: the gate — learns to suppress or amplify individual dimensions
- Multiplied element-wise: adaptive, input-dependent non-linearity

Adds a third weight matrix $W_3$ but FFN width is typically reduced to keep total FLOPs constant. Empirically outperforms GELU across the board.

> **Why SwiGLU Uses Hidden Width $\frac{8}{3} d_{\text{model}}$ Instead of $4 d_{\text{model}}$:**
> - A standard 2-matrix FFN ($W_1 \in \mathbb{R}^{d \times 4d}$, $W_2 \in \mathbb{R}^{4d \times d}$) has $2 \times (d \times 4d) = \mathbf{8 d^2}$ parameters.
> - SwiGLU has **3 matrices** ($W_1, W_3 \in \mathbb{R}^{d \times d_{\text{ff}}}$ and $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$), totaling $3 \times d \times d_{\text{ff}}$ parameters.
> - Setting $d_{\text{ff}} = \frac{2}{3}(4d) = \mathbf{\frac{8}{3} d}$ makes the total parameter count $3 \times d \times \frac{8}{3}d = \mathbf{8 d^2}$ — matching the exact parameter and FLOP budget of a standard $4d$ FFN!

### Summary

| Activation | Used in | Key property |
|---|---|---|
| ReLU | Original Transformer (2017) | Simple; dying neuron problem |
| GELU | BERT, GPT-2, GPT-3 | Smooth; no hard zero |
| SwiGLU | LLaMA, Mistral, PaLM | Gated; adaptive; modern default |

---
