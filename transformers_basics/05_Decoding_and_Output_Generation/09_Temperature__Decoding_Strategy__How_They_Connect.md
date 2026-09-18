## 8. Temperature + Decoding Strategy — How They Connect

Temperature and decoding strategy are **two separate knobs**, but they interact:

```
  Temperature reshapes the distribution.
  Decoding strategy decides how to pick from it.
```

### Temperature is a no-op with Greedy or Beam Search

Greedy and beam search are both **deterministic** — they always pick the highest-scoring token(s).
Temperature never reorders tokens (rank #1 stays rank #1 at any T), so the output is identical regardless of T.

```
  Logits:  "the"=3.2  "a"=1.8  "an"=1.5  ...

  T=0.5 → P: [0.900, 0.055, 0.030, ...]   argmax = "the"
  T=1.0 → P: [0.621, 0.153, 0.113, ...]   argmax = "the"
  T=2.0 → P: [0.396, 0.197, 0.169, ...]   argmax = "the"
                                            ↑ same every time
  → Setting temperature on greedy/beam is a no-op.
```

### Temperature only matters with Sampling

Sampling **draws** from the distribution — so the shape directly affects the outcome.

```
  T=0.5  → distribution very peaked  → almost always "the"   (safe, repetitive)
  T=1.0  → balanced                  → mostly "the", sometimes others
  T=2.0  → distribution flat         → "the", "a", "an" all competitive (creative, risky)
```

### Full Picture

```
  ┌──────────────┬───────────────────────────────────────────────────────┐
  │              │               Decoding Strategy                       │
  │              ├───────────────┬───────────────┬───────────────────────┤
  │              │    Greedy     │  Beam Search  │  Sampling (top-k/p)   │
  ├──────────────┼───────────────┼───────────────┼───────────────────────┤
  │ Temperature  │               │               │                       │
  │   T < 1      │   no effect   │   no effect   │  more deterministic   │
  │   T = 1      │   no effect   │   no effect   │  default behaviour    │
  │   T > 1      │   no effect   │   no effect   │  more diverse/random  │
  └──────────────┴───────────────┴───────────────┴───────────────────────┘
```

> **Bottom line:** Temperature is only meaningful when paired with a sampling strategy. Greedy and beam search ignore it entirely.

---
