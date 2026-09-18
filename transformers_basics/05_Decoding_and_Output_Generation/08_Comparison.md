## 7. Comparison

| Strategy | Deterministic | Diversity | Quality | Speed |
|---|---|---|---|---|
| Greedy | Yes | None | Low (local optima) | Fastest |
| Beam Search | Yes | Low | Better than greedy | Slow (B× cost) |
| Top-k Sampling | No | Medium | Good | Fast |
| Top-p Sampling | No | Adaptive | Best in practice | Fast |

**Common production default:** Top-p (nucleus) sampling with temperature, often combined:

$$\text{logits} \xrightarrow{÷ T} \text{scaled logits} \xrightarrow{\text{top-p filter}} \text{nucleus} \xrightarrow{\text{sample}} \text{next token}$$

---
