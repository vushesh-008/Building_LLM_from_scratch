## 11. RoPE — Rotary Positional Embeddings

> **Default choice in modern LLMs:** LLaMA, Mistral, Gemma, GPT-NeoX

**Core idea:** Instead of adding a position vector to token embeddings, RoPE **rotates** the Q and K vectors inside the attention layer by an angle proportional to the token's position. The token embedding itself is never touched — only the queries and keys used in the dot product are rotated.

```
  Sinusoidal / learned PE:   token_vec = embedding + position_vec   ← modifies embedding
  RoPE:                      q̃ = R(mθ) · q   ,   k̃ = R(nθ) · k   ← only inside attention
```

---

### Step 1 — Rotate, don't add (one 2D pair)

RoPE operates on **pairs of dimensions** at a time. For a single 2D pair $(x, y)$ belonging to a query or key at position $m$, it applies a rotation matrix:

$$R(m\theta) = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}$$

- Position 0 → rotate by $0$ (no change)
- Position 1 → rotate by $\theta$
- Position 2 → rotate by $2\theta$
- Position $m$ → rotate by $m\theta$

The vector's **magnitude never changes** — only its direction shifts. This matters: rotation preserves the token's content signal, it only stamps "where."

![RoPE rotates a 2D dimension pair by an angle proportional to position](images/rope_single_pair_rotation.svg)

> Each coloured arrow is the **same vector**, just spun by a larger angle. Position is encoded as direction, not added noise.

---

### Step 2 — Full embedding: many pairs, many frequencies

A real embedding (e.g. $d=128$) is split into **64 separate 2D pairs**. Each pair gets its own rotation frequency $\theta_i$:

$$\theta_i = \text{base}^{-2i/d} \qquad \text{base} = 10{,}000$$

| Pair index $i$ | $\theta_i$ | Rotation speed | Analogy |
|---|---|---|---|
| 0 (dims 0–1) | large | very fast | seconds hand |
| 15 (dims 30–31) | medium | moderate | minute hand |
| 31 (dims 62–63) | tiny | extremely slow | hour hand |

![Different dimension pairs rotate at different frequencies](images/rope_multi_frequency_pairs.svg)

**Why multiple frequencies?**

- **Fast pairs** (low $i$): rotate a lot even between neighbouring tokens — sensitive to fine local position differences
- **Slow pairs** (high $i$): barely move even across thousands of tokens — encode coarse, long-range position without aliasing

This is the same idea as sinusoidal PE's multi-frequency design, but applied as rotation inside attention rather than addition to embeddings.

```
  For d = 128, at position m = 1000:

  Pair 0:   rotates by  1000 × θ₀  ≈  many full turns   ← fine-grained
  Pair 31:  rotates by  1000 × θ₃₁ ≈  tiny fraction     ← coarse
```

---

### Step 3 — Relative position falls out for free

This is the key payoff. When attention computes $\tilde{q}_m \cdot \tilde{k}_n$, the rotation matrices combine:

$$\tilde{q}_m \cdot \tilde{k}_n^\top = q_m \cdot R(\theta m)^\top R(\theta n) \cdot k_n^\top = q_m \cdot R\!\left(\theta(n-m)\right) \cdot k_n^\top$$

The absolute positions $m$ and $n$ cancel — only the **relative distance** $(n-m)$ survives in the dot product.

![Same relative offset produces the same angle regardless of absolute position](images/rope_relative_position_invariance.svg)

Both pairs in the diagram (positions 3 & 7, and positions 0 & 4) have offset 4 and both produce $\Delta = 80°$ between their rotated vectors. Different absolute positions, identical relative signal — the model can never tell whether it's looking at tokens (0, 4) or (3, 7), only that they are 4 apart.

**Why this matters for generalisation:** the model never memorises "position 50,000 means X." It only learns "tokens $k$ apart interact like this" — a relationship baked directly into the rotation math. This is what allows RoPE-based models to extrapolate to longer sequences than seen during training (with appropriate scaling tricks like YaRN, LongRoPE).

### Summary — RoPE vs the alternatives

| | Sinusoidal | Learned absolute | ALiBi | RoPE |
|---|---|---|---|---|
| Added to embedding? | Yes | Yes | No | No |
| Where applied | Input | Input | Attention scores | Q, K inside attention |
| Relative position? | Partial (sin+cos trick) | No | Yes (linear bias) | Yes (exact) |
| Extrapolates? | Poorly | No | Yes | Yes (with scaling) |
| Used in | Original Transformer | GPT-2/3, BERT | BLOOM, MPT | LLaMA, Mistral, Gemma |

---
