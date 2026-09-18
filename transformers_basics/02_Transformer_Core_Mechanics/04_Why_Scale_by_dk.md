## 3. Why Scale by √dk?

Before writing out the full self-attention formula, we need to understand why the raw scores are divided by $\sqrt{d_k}$.

### The problem — dot products grow with dimension

Each raw score is a dot product of a query and a key:

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

If each component is drawn from a distribution with mean 0 and variance 1:

$$\text{Var}(q \cdot k) = d_k$$

The variance of the dot product **scales linearly with $d_k$**. For $d_k = 64$, the typical dot product magnitude is around $\sqrt{64} = 8$.

### What this does to softmax

Softmax is exponential — it amplifies differences:

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

When scores are large (e.g. $[8, -2, 0, 1]$), softmax becomes nearly one-hot — almost all weight goes to the largest score, gradients for all other positions approach zero, and training stalls.

```
  d_k = 64, raw scores ≈ [8.0, -2.0, 0.0, 1.0]
  softmax → [0.9997, 0.0000, 0.0001, 0.0002]   ← near one-hot, dead gradients

  After ÷ √64 = 8 → scaled scores ≈ [1.0, -0.25, 0.0, 0.125]
  softmax → [0.47, 0.18, 0.23, 0.26]            ← smooth distribution ✓
```

### The fix

$$S_{\text{scaled}} = \frac{QK^\top}{\sqrt{d_k}} \qquad \Rightarrow \quad \text{Var}(S_{\text{scaled}}) = 1$$

Dividing by $\sqrt{d_k}$ brings variance back to 1 regardless of dimension. The formula that appears in every transformer paper:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

---
