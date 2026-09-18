## 3. Text Embedding vs Patch Embedding

This is the first fundamental difference. How a token enters the model depends on whether it's a word or a pixel patch.

### Text tokens — embedding is retrieval

Text tokens are discrete integers (vocabulary IDs). Their embedding is a **lookup** — index into a table and retrieve a row.

```
  Vocabulary table (learned):
  ┌──────┬──────────────────────────────────────┐
  │  ID  │  Embedding vector [d_model]           │
  ├──────┼──────────────────────────────────────┤
  │    0 │  [0.12, -0.34, 0.87, ...]            │  ← <pad>
  │    1 │  [0.55,  0.21, -0.09, ...]           │  ← <eos>
  │  ...                                        │
  │ 2541 │  [-0.18, 0.63, 0.44, ...]            │  ← "cat"
  │  ...                                        │
  └──────┴──────────────────────────────────────┘

  Token "cat" (ID 2541) → retrieve row 2541 → [−0.18, 0.63, 0.44, ...]
  No computation needed. Just a memory read.
```

### Image patches — embedding is computation

Pixel patches have no discrete ID. A 14×14 RGB patch is 588 continuous float values. Their embedding is a **linear projection** — multiply the raw pixels by a learned weight matrix.

```
  Image patch  (14×14×3 = 588 pixel values):
  [0.91, 0.43, 0.12, 0.88, ..., 0.37]   ← 588 floats (raw pixels)
         │
         ▼  × W_patch  [588 × d_vit]
         │
  [0.24, -0.11, 0.67, ...]              ← d_vit dimensional patch embedding
  (learned linear layer, not a lookup)
```

| | Text tokens | Image patches |
|---|---|---|
| Input type | Discrete integer (token ID) | Continuous floats (pixel values) |
| Embedding method | Table lookup (retrieval) | Linear projection (computation) |
| Learned parameters | Embedding matrix rows | Weight matrix W_patch |
| Output | [d_model] vector | [d_vit] vector |

> **Key insight:** Text embedding is retrieval — it maps a symbol to a vector. Patch embedding is computation — it maps a signal to a vector. The rest of the pipeline doesn't care which it was.

---
