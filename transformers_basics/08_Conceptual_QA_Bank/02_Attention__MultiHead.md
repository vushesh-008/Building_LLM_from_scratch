## Attention & Multi-Head

---

**Q: Why scale attention scores by $\sqrt{d_k}$? What goes wrong without it?**

Each attention score is a dot product of a query and a key: $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$. If the elements are roughly unit normal, the variance of this sum is $d_k$ — so the standard deviation grows as $\sqrt{d_k}$.

With $d_k = 768$, scores can easily reach magnitudes of ~28 before softmax. Softmax of large values is extremely peaked — one entry gets probability ≈ 1 and all others ≈ 0. In this saturated regime the gradients of softmax are nearly zero (the function is flat), and learning stalls.

Dividing by $\sqrt{d_k}$ brings the variance back to 1, keeping scores in a range where softmax has meaningful gradients:

```
  Without scaling (d_k = 768):  scores ~ N(0, 28)  →  softmax saturates  →  vanishing gradients
  With scaling:                  scores ~ N(0, 1)   →  softmax healthy    →  learning works
```

---

**Q: Why does self-attention have no notion of order without positional encoding?**

The attention score between token $i$ and token $j$ is computed purely from their content vectors: $s_{ij} = q_i \cdot k_j^T$. There is no index, no notion of "before" or "after" — just two vectors and their dot product.

If you shuffle the input sequence, the same attention scores are produced (for the same pairs), just in a different arrangement. The output of attention is a weighted sum of value vectors — that weighted sum is the same regardless of the order the tokens arrived in. Self-attention is, by design, **permutation equivariant**. Positional encoding breaks this symmetry by injecting order information into the vectors before attention sees them.

---

**Q: Why are Q heads never shared in GQA/MQA, even though K and V are?**

The whole purpose of multi-head attention is that different heads attend to different things — one head might track syntactic dependencies, another co-reference, another local context. What makes each head different is its query projection: $Q_i = x \cdot W_{Q_i}$. Each head asks a different question of the sequence.

Sharing Q heads would force all heads to ask the identical question — they'd produce the same attention weights and the same output, making multiple heads pointless.

K and V can be shared because the *information available* to answer queries doesn't need to be multiplied — one set of keys and values contains the full sequence content. Different Q heads can query the same K/V from different angles and still attend to different parts of the sequence.

```
  Share K/V: multiple perspectives on the same information  ✓  (this is the point)
  Share Q:   multiple heads asking the same question        ✗  (defeats the purpose)
```

---

**Q: Why is the FFN per-token but attention is cross-token — and why does that matter for MoE?**

Attention computes relationships between tokens — token $i$'s output depends on tokens $j, k, l...$ via the key-query dot products. You cannot process one token independently of others in the attention layer.

The FFN that follows processes each token's representation completely independently — the same function is applied to token 1, then token 2, then token 3, with no information flowing between them. Token $i$'s FFN output depends only on token $i$'s input.

This independence is what makes MoE possible in the FFN layer. Because each token is processed separately, you can route each token to a different expert without any cross-token communication issues. If you tried to put MoE in the attention layer, different tokens attending to each other through different experts would create a consistency problem — Expert A's keys would need to be compatible with Expert B's queries.

---
