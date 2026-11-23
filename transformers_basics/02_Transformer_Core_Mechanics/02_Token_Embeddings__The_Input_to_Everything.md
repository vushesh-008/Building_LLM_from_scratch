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

### Why Embedding Lookup Is Mathematically Identical to One-Hot Matrix Multiplication
In *Build a Large Language Model from Scratch* (Ch. 2), Sebastian Raschka highlights an important equivalence:
- Suppose vocabulary size $V = 4$ and $d_{\text{model}} = 3$, and we look up token ID `2`.
- Represent token ID `2` as a **one-hot row vector** $\mathbf{o}_2 = [0, 0, 1, 0] \in \mathbb{R}^{1 \times 4}$.
- Multiply $\mathbf{o}_2$ by the embedding weight matrix $E \in \mathbb{R}^{4 \times 3}$:
  $$[0, \; 0, \; 1, \; 0] \cdot \begin{bmatrix} e_{00} & e_{01} & e_{02} \\ e_{10} & e_{11} & e_{12} \\ \mathbf{e_{20}} & \mathbf{e_{21}} & \mathbf{e_{22}} \\ e_{30} & e_{31} & e_{32} \end{bmatrix} = [\mathbf{e_{20}}, \; \mathbf{e_{21}}, \; \mathbf{e_{22}}] \quad (\text{Row 2 of } E!)$$
- Thus, `nn.Embedding(V, d_model)` is mathematically identical to a `nn.Linear(V, d_model, bias=False)` layer fed with one-hot vectors — except `nn.Embedding` skips multiplying by all the `0`s and directly fetches Row $i$ in $O(1)$ memory lookup!

### Weight Tying (Sharing Input Embeddings with the Output LM Head)
Many LLMs (including GPT-2, LLaMA-3 8B, and Gemma) **tie (share)** the input embedding matrix $E \in \mathbb{R}^{V \times d_{\text{model}}}$ with the final output projection head $W_{\text{out}} = E^\top \in \mathbb{R}^{d_{\text{model}} \times V}$.
- **Why?** It saves hundreds of millions of parameters (e.g., $50,257 \times 768 \approx 38.6\text{M}$ params in GPT-2) and enforces a clean symmetry: tokens with similar input embeddings also receive similar output logit scores.

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
