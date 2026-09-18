## 1. How Blocks Stack — Shape-Preserving Property

The entire reason you can stack $N$ identical blocks is that every block is **shape-preserving**:

$$[N_{seq} \times d_{model}] \xrightarrow{\text{Block}} [N_{seq} \times d_{model}]$$

Same shape in, same shape out. Block 2 cannot tell whether its input came from raw embeddings or from 50 prior blocks — it just sees a matrix of the right shape and refines it.

![Shape-preserving residual stream through stacked transformer blocks](images/residual_stream_shape_preserving.svg)

```
  Embeddings  [Seq × 512]
       │
  Block 1     [Seq × 512] → [Seq × 512]   ← same shape in and out
       │
  Block 2     [Seq × 512] → [Seq × 512]
       │
      ...
       │
  Block N     [Seq × 512] → [Seq × 512]
       │
  Output      [Seq × 512]
```

![Sequential transformer block stacking](images/sequential_block_stack_vertical.svg)

The tensor gets **refined**, never reshaped. Empirically: early blocks capture local patterns and syntax; later blocks capture long-range semantics and reasoning. The architecture doesn't enforce this — it emerges from training.

---
