## Decoding

---

**Q: Why is temperature a complete no-op when using greedy decoding?**

Greedy decoding always picks $\arg\max P$ — the single highest-probability token. Temperature rescales the logits before softmax ($\text{logit} / T$), which changes the probability values but never changes their *ranking*. If token A had a higher logit than token B before dividing by $T$, it still has a higher logit after dividing by $T$ (for any $T > 0$).

```
  Logits:  "the"=3.2,  "a"=1.8,  "an"=1.5

  T=0.5:  P = [0.900, 0.055, 0.030]   argmax = "the"
  T=1.0:  P = [0.621, 0.153, 0.113]   argmax = "the"
  T=2.0:  P = [0.396, 0.197, 0.169]   argmax = "the"
```

Temperature only matters when you **sample** from the distribution — because sampling is sensitive to the *shape* of the distribution, not just its argmax.

---

**Q: Why does top-p adapt to the shape of the distribution while top-k doesn't?**

Top-k always keeps exactly $k$ tokens regardless of the distribution's shape. When the model is very confident (one token at 0.95 probability), top-k=50 still forces you to sample from 50 tokens — 49 of which are near-zero garbage. When the model is uncertain (20 tokens each at ~0.05), top-k=3 throws away most valid options.

Top-p keeps the smallest set of tokens whose cumulative probability reaches $p$. When the model is confident, that set is small (maybe 1–2 tokens). When the model is uncertain, that set is large (maybe 20+ tokens). The nucleus automatically contracts and expands with the distribution:

```
  Peaked distribution ("the" → 0.92):
    top-k=50  →  keeps 49 near-zero tokens (bad)
    top-p=0.9 →  nucleus = {"the"} only    (good)

  Flat distribution (20 tokens × 0.05):
    top-k=3   →  discards 17 valid tokens  (bad)
    top-p=0.9 →  nucleus = 18 tokens       (good)
```

---

**Q: Why does beam search still produce generic/repetitive output despite exploring multiple paths?**

Beam search maximises the joint probability of the full sequence — it finds the sequence most likely under the model. But "most likely" under a language model trained on diverse human text often means the safest, most average, most seen-in-training sequence. It's the mode of the distribution.

Human language is not mode-seeking — people naturally produce diverse, creative, non-generic text. By explicitly hunting for the highest-probability path, beam search systematically avoids anything surprising or diverse. It also tends to repeat phrases, because repetition has high conditional probability given what was just said.

Sampling-based methods avoid this by not optimising for the mode — they draw from the distribution and so naturally produce more varied output.

---
