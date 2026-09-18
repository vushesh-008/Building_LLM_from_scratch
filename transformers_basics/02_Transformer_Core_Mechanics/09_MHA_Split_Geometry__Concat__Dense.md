## 6. MHA Split Geometry — Concat + Dense

Each head operates on a $d_k$-dimensional **projection** of the full input, not a slice.

```
  Input X  [N × 512]
       │
       ├─────────────────────────────────────────────────┐
       │  W_Q¹,W_K¹,W_V¹  [512→64]   W_Qʰ,W_Kʰ,W_Vʰ  [512→64]
  ┌──────────┐                                    ┌──────────┐
  │  Head 1  │  softmax(Q₁K₁ᵀ/√64)·V₁           │  Head 8  │
  │  [N×64]  │                                    │  [N×64]  │
  └────┬─────┘                                    └────┬─────┘
       │                                               │
       └───────────────── CONCAT ────────────────────┘
                              │
                       [N × 512]   (64 × 8 = 512)
                              │
                       × W_O  [512 × 512]   ← Dense
                              │
                       Output [N × 512]
```

**Why W_O matters:** Without it, head 1's 64-dim output and head 2's 64-dim output would sit in permanently isolated subspaces — they'd never interact. $W_O$ is the only step where information flows across heads.

![Multi-head attention splitting and recombination](images/multihead_attention_split_geometry.svg)

### Parameter count (d_model=512, h=8, d_k=64)

| Weight | Shape | Parameters |
|---|---|---|
| $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$ per head | 512 × 64 each | 32,768 × 3 = 98,304 |
| All 8 heads | — | 8 × 98,304 = **786,432** |
| $W_O$ output projection | 512 × 512 | **262,144** |
| **Total MHA** | | **~1.05 M** |

This is the same parameter count as a single-head attention with full 512-dim Q, K, V. Multi-head doesn't add parameters — it redistributes them across specialised subspaces.

---
