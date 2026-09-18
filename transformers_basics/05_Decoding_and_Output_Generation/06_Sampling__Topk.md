## 5. Sampling — Top-k

Instead of argmax, **randomly sample** from the distribution — but only from the **top-k tokens**.

```
  Full distribution (vocab = 50K tokens):

  "the"  0.35  ┐
  "a"    0.22  ├── Top-3 (k=3) → keep these, renormalize
  "an"   0.18  ┘
  "one"  0.10  ← cut off
  "any"  0.08  ← cut off
   ...   ...   ← cut off (all ~50K others)

  Renormalized over top-3:
  "the"  0.35 / 0.75 = 0.467
  "a"    0.22 / 0.75 = 0.293
  "an"   0.18 / 0.75 = 0.240

  → Sample from {the: 0.467, a: 0.293, an: 0.240}
```

**Pros:** Introduces diversity; avoids the very long tail of garbage tokens.

**Cons:** k is fixed regardless of how peaked/flat the distribution is. With a very flat distribution, top-3 might be too restrictive. With a very peaked one, k=50 might include junk.

---
