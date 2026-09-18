## 2. Temperature Scaling

Before softmax, logits are divided by a **temperature** $T$:

$$P(token_i) = \frac{\exp(logit_i / T)}{\sum_j \exp(logit_j / T)}$$

Temperature controls **how peaked or flat** the distribution is.

### Starting Point — Raw Logits

The model outputs raw logits (unnormalised scores) for each vocabulary token. Let's use 5 tokens:

```
  Token    Logit
  ──────   ─────
  "the"     3.2
  "a"       1.8
  "an"      1.5
  "one"     0.9
  "any"     0.7
```

These logits get divided by $T$ before softmax. Watch what happens.

---

### T = 1.0 — Default (no scaling)

$$logit / T = logit / 1.0 = logit \quad \text{(unchanged)}$$

```
  Token    Logit    exp(logit)    P = exp / Σexp
  ──────   ─────    ──────────    ──────────────
  "the"     3.2       24.53          0.621   ██████████████████████████
  "a"       1.8        6.05          0.153   ██████
  "an"      1.5        4.48          0.113   ████
  "one"     0.9        2.46          0.062   ██
  "any"     0.7        2.01          0.051   ██
                    ────────       ──────
             Σ =    39.53          1.000
```

"the" dominates but the other tokens still have a real chance.

---

### T = 0.5 — Low Temperature (sharper)

$$logit / T = logit / 0.5 \quad \text{(scaled UP — gaps between logits grow)}$$

```
  Token    Logit    /T=0.5    exp(logit/T)    P = exp / Σexp
  ──────   ─────    ──────    ────────────    ──────────────
  "the"     3.2      6.40        601.85          0.900   ████████████████████████████████████
  "a"       1.8      3.60         36.60          0.055   ██
  "an"      1.5      3.00         20.09          0.030   █
  "one"     0.9      1.80          6.05          0.009   ░
  "any"     0.7      1.40          4.05          0.006   ░
                               ────────         ──────
                        Σ =    668.64           1.000
```

Low T amplifies differences — "the" goes from 62% → **90%**. The tail nearly vanishes.

---

### T = 2.0 — High Temperature (flatter)

$$logit / T = logit / 2.0 \quad \text{(scaled DOWN — gaps between logits shrink)}$$

```
  Token    Logit    /T=2.0    exp(logit/T)    P = exp / Σexp
  ──────   ─────    ──────    ────────────    ──────────────
  "the"     3.2      1.60         4.953          0.396   ████████████████
  "a"       1.8      0.90         2.460          0.197   ████████
  "an"      1.5      0.75         2.117          0.169   ███████
  "one"     0.9      0.45         1.568          0.125   █████
  "any"     0.7      0.35         1.419          0.113   █████
                               ────────         ──────
                        Σ =    12.517           1.000
```

High T compresses differences — "the" drops from 62% → **40%**. Other tokens become much more competitive.

---

### Side-by-side Summary

```
  Token      T=0.5    T=1.0    T=2.0
  ──────     ─────    ─────    ─────
  "the"      0.900    0.621    0.396   ← always most likely, but less dominant as T↑
  "a"        0.055    0.153    0.197
  "an"       0.030    0.113    0.169
  "one"      0.009    0.062    0.125
  "any"      0.006    0.051    0.113
             ─────    ─────    ─────
             1.000    1.000    1.000
```

> The raw **ranking never changes** — temperature doesn't reorder tokens, it only reshapes how much probability mass is concentrated at the top.

| Temperature | Effect | Use case |
|---|---|---|
| T → 0 | Argmax — always picks the top token | Deterministic / factual tasks |
| T = 1 | Default softmax | General use |
| T > 1 | Flatter distribution | Creative writing, diversity |
| T → ∞ | Uniform distribution | Pure random |

---
