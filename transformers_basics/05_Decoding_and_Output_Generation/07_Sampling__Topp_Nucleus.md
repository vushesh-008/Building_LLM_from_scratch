## 6. Sampling — Top-p (Nucleus)

Instead of fixing k tokens, fix a **cumulative probability threshold p** and sample from the smallest set of tokens whose cumulative probability ≥ p.

```
  p = 0.90

  Sort tokens by probability (descending):
  ┌────────┬──────┬────────────┐
  │ Token  │  P   │ Cumul. P   │
  ├────────┼──────┼────────────┤
  │ "the"  │ 0.35 │ 0.35       │
  │ "a"    │ 0.22 │ 0.57       │
  │ "an"   │ 0.18 │ 0.75       │
  │ "one"  │ 0.10 │ 0.85       │
  │ "any"  │ 0.08 │ 0.93  ◄── crosses 0.90 here
  ├────────┼──────┼────────────┤
  │ "big"  │ 0.03 │ 0.96       │ ← cut off
  │  ...   │ ...  │ ...        │ ← cut off
  └────────┴──────┴────────────┘

  Nucleus = {"the", "a", "an", "one", "any"}
  Renormalize and sample from these 5 tokens only.
```

### Why Top-p > Top-k

```
  Peaked distribution (model is confident):
  "the" → 0.92, rest tiny

    Top-k=50 would include 49 garbage tokens
    Top-p=0.90 → nucleus = {"the"} only  ✓

  Flat distribution (model is uncertain):
  20 tokens each ~0.05

    Top-k=3 would throw away most valid options
    Top-p=0.90 → nucleus = 18 tokens  ✓

  Top-p adapts to the shape of the distribution. Top-k does not.
```

---
