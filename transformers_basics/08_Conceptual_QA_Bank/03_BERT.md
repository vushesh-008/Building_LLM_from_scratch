## BERT

---

**Q: Why does BERT use 80/10/10 for masking instead of always replacing with [MASK]?**

If 100% of selected tokens were always replaced with [MASK], the model would learn that [MASK] tokens need to be predicted but real tokens don't — at fine-tuning time there are no [MASK] tokens, so the model's representations for real tokens would be undertrained.

The 10% random replacement forces the model to maintain good representations for every token position, because any token could be the "wrong" word that needs to be corrected. The 10% unchanged forces the model to keep the original token's representation useful, because sometimes the "answer" is just the token itself. Together they close the gap between pre-training (lots of [MASK]) and fine-tuning (no [MASK]).

```
  80% → [MASK]:   model must predict from context          (main learning signal)
  10% → random:   model must detect and correct errors     (no free ride on real tokens)
  10% → unchanged: model must copy/confirm the real word   (representation stays useful)
```

---

**Q: Why is [CLS] placed at the *beginning* rather than the end?**

BERT uses bidirectional attention — every token attends to every other token in both directions. So [CLS] sees the full sequence regardless of whether it's at position 0 or position N.

The practical reason is causal: in the original Transformer decoder (and for compatibility with later causal models), processing flows left to right, so putting the classification token at the start means it accumulates information from the full sequence through attention by the time it reaches the final layer. It also makes engineering simpler — the classification head always reads from index 0 of the output, regardless of sequence length.

---

**Q: Why does BERT train on sequences of 128 tokens for 90% of steps, then 512 for only 10%?**

Attention is $O(N^2)$ in sequence length — training on 512-token sequences costs $(512/128)^2 = 16\times$ more compute per step than 128-token sequences.

Most of what BERT needs to learn (word-level semantics, local syntactic patterns, MLM) can be learned efficiently on shorter sequences. The final 10% on 512-token sequences teaches the model to use its positional embeddings at longer ranges and handle longer-range dependencies — but this requires far fewer steps because the shorter-context knowledge already built up. The schedule gives 90% of the learning cheaply, then buys long-context ability efficiently at the end.

---

**Q: Why pre-norm (modern) vs post-norm (original paper)?**

**Post-norm** (original): $\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))$

The residual connection adds the raw sublayer output before normalising. At the start of training, sublayer outputs can be large and noisy, and the raw residual compounds this — deep post-norm networks are notoriously hard to train without careful learning rate warmup.

**Pre-norm** (modern): $\text{output} = x + \text{SubLayer}(\text{LayerNorm}(x))$

The input is normalised *before* the sublayer. The residual path ($x$) is always clean — the unnormalised signal flows directly through the skip connection. This keeps gradients well-behaved at any depth and makes training stable without warmup tricks.

```
  Post-norm: noisy signal → sublayer → add residual → normalize  (hard to train deep)
  Pre-norm:  clean signal → normalize → sublayer → add residual   (stable at depth)
```

Most modern models (LLaMA, GPT-NeoX) use pre-norm + RMS norm, which drops the mean-subtraction step for further efficiency.

---
