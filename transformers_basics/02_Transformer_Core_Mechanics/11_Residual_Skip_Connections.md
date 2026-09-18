## 8. Residual Skip Connections

Every sublayer (attention or FFN) has a **bypass wire** that carries the input around it completely untouched. The output is the sublayer's result added back to that original input:

$$\text{block output} = \text{LayerNorm}(X + \text{Sublayer}(X))$$

The $X$ in that formula traveled around `Sublayer` unchanged — that is the skip connection.

![Explicit residual skip connections around each sublayer](images/residual_skip_connection_explicit.svg)

### What "Add & Norm" means in diagrams

You'll see "Add & Norm" in every transformer diagram. It means exactly:
1. **Add**: element-wise add the sublayer output to the original input ($X + \text{Sublayer}(X)$)
2. **Norm**: apply Layer Normalisation to the sum

### Why residuals are essential for training

During backpropagation, the `+` operation passes gradients **directly through unchanged** — no multiplication, no squashing. Even if the sublayer has near-zero or saturated gradients, the skip gives gradients a clean path all the way back to the embedding layer.

```
  Without residuals (depth 96):
  Gradient at layer 1 ≈ (small number)^96 ≈ 0   ← vanished

  With residuals:
  Gradient at layer 1 = (gradient from layer 2) + (direct skip gradient)
                      ≈ anything + something ≠ 0   ← training works
```

**Early training intuition:** At initialisation, sublayer weights are near-random. Without residuals, the signal degrades through depth. With residuals, the model starts as a near-identity function — each sublayer adds a small learned correction on top of a clean signal — and gradually refines it.

---
