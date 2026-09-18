## 1. Token Embeddings — The Input to Everything

Before any attention can happen, raw text must become vectors. This is the very bottom of the stack.

### Tokens → integers

A tokenizer (e.g. WordPiece, BPE) maps each word or subword to an integer ID from a fixed vocabulary:

```
  "The cat sat"
      │
  Tokenizer (vocab size ~30K–100K)
      │
  [101, 2482, 4429, ...]    ← integer token IDs
```

### Integers → vectors (the embedding lookup)

An **embedding table** $E \in \mathbb{R}^{V \times d_{model}}$ holds one learned row per vocabulary entry. Looking up token $i$ is just retrieving row $i$:

```
  Embedding table  (V rows × d_model cols):
  ┌──────┬──────────────────────────────────┐
  │  ID  │  Embedding vector [d_model]       │
  ├──────┼──────────────────────────────────┤
  │    0 │  [0.12, -0.34,  0.87, ...]       │  ← <pad>
  │  101 │  [0.55,  0.21, -0.09, ...]       │  ← [CLS]
  │ 2482 │  [-0.18, 0.63,  0.44, ...]       │  ← "cat"
  │  ...                                    │
  └──────┴──────────────────────────────────┘

  Token ID 2482 → retrieve row 2482 → [-0.18, 0.63, 0.44, ...]
  No computation — just a memory read.
```

### What the embedding table contains at each stage

| Stage | What the rows contain |
|---|---|
| Randomly initialised (day 0) | Small random numbers — no meaning yet |
| After pre-training | Vectors that encode semantic similarity — "cat" and "dog" are nearby |
| Frozen at fine-tuning | Fixed; only task-specific heads update |

The embedding table is updated by backpropagation just like any weight matrix.

### Shape summary

```
  N tokens
      │
      ▼  E  [V × d_model]   (index into table)
      │
  [N × d_model]   ← this matrix is the input to all transformer blocks
```

This $[N \times d_{model}]$ matrix — one $d_{model}$-dimensional vector per token — is what flows into the attention mechanism. Every section from here onward assumes this as input.

---
