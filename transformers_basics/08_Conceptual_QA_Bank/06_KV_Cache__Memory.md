## KV Cache & Memory

---

**Q: Why does training not need a KV cache but inference does?**

During training the full sequence is available at once. All token positions are processed in parallel with causal masking — position 5 can attend to positions 1–4 via the mask without needing to store and retrieve their K/V from a cache. Everything is computed in one forward pass.

During inference the model generates one token at a time, autoregressively. To predict token $t$, it needs K and V for all previous tokens $1$ to $t-1$. Without caching, those K and V vectors would need to be recomputed from scratch at every step — O(N²) total work. The KV cache stores them once and reuses them.

---

**Q: Why does contiguous memory allocation force a full deep copy when beam search forks?**

When beam search creates two branches from the same prompt, both branches will append *different* future tokens to their KV buffers. A single contiguous array cannot be shared between them — if Branch A writes token "left" at position 10, it overwrites whatever Branch B would have written there.

So the server must physically copy the entire prompt's KV tensor into two separate contiguous memory regions before either branch can safely write its next token. With k=3 beams, that's 3 full copies of the prompt KV at fork time — even though the data is identical.

PagedAttention solves this with copy-on-write: all beams share the same physical blocks until one needs to *write* to a shared block, at which point only that one block is copied, not the entire sequence.

---

**Q: Why does PagedAttention's copy-on-write only ever touch the *last partial block* and not earlier shared blocks?**

Once a block is completely full (all $B$ token slots written), it will never be written to again — there are no free slots. A completely full shared block can be read by all beams simultaneously with no risk of any beam overwriting it.

The only block that ever needs a write is the *current* block — the one where new tokens are being appended. At the moment beams diverge, only this last (potentially partial) block is in danger of being written to differently by different beams. Earlier blocks are sealed.

```
  Blocks #1, #2, #3: full (4/4 tokens) → read-only, safe to share forever
  Block #4: partial (2/4 tokens used)  → beams will write different tokens here
             → only this one gets copy-on-write at divergence
```

---

**Q: With GQA (G groups), what is the KV cache memory reduction relative to MHA?**

In MHA with $h$ heads, you cache $h$ K-head vectors and $h$ V-head vectors per token per layer.
In GQA with $G$ groups, you cache $G$ K-head vectors and $G$ V-head vectors.

Reduction factor: $h / G$

```
  h = 32 heads (typical large model):

  G = 32 (MHA):   32 K + 32 V heads cached  →  1× (baseline)
  G = 8  (GQA):    8 K +  8 V heads cached  →  4× smaller KV cache
  G = 1  (MQA):    1 K +  1 V head  cached  →  32× smaller KV cache
```

LLaMA 3 70B uses $h=64$ query heads, $G=8$ KV groups → 8× KV cache reduction vs full MHA, with quality very close to MHA.

---
