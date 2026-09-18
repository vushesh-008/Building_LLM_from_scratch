## Positional Encoding

---

**Q: Why does the sinusoidal PE formula use sin/cos at different frequencies rather than just position indices?**

Plain position indices (0, 1, 2, 3...) would work as a signal, but they have two problems: they grow unboundedly for long sequences, and the model has to learn to interpret raw integers. Sinusoidal functions solve both — they are bounded between –1 and +1 regardless of sequence length, and each dimension oscillates at a different frequency, so every position gets a unique fingerprint across the full embedding vector.

The different frequencies also matter geometrically: low-frequency dimensions change slowly across positions (capturing coarse, long-range structure) while high-frequency dimensions change rapidly (capturing fine, local structure). The model can read off both coarse and fine position information from the same vector.

```
  dim 0 (high freq):  sin(1·m),  sin(2·m),  sin(3·m) ...  changes fast
  dim 512 (low freq): sin(0.001·m) ...                     changes slowly

  → Together they give a unique, stable fingerprint for every position m
```

---

**Q: Why does the dot product of two sinusoidal PEs depend only on their *relative* distance and not absolute positions — and why is that desirable?**

Using $\cos(a - b) = \cos a \cos b + \sin a \sin b$, the dot product of $PE_m$ and $PE_n$ simplifies to a sum of $\cos(\omega_i(m - n))$ terms. The result is a function of $(m - n)$ only — the absolute positions $m$ and $n$ cancel out.

This is desirable because language meaning is usually about *relative* relationships ("the word 3 positions back") rather than absolute positions ("word number 47 in the document"). A model that encodes relative distance generalises better — the pattern "subject is 2 tokens before verb" works whether the subject is at position 5 or position 500.

---

**Q: Why does RoPE rotate Q and K *inside* the attention layer rather than adding to the embeddings like sinusoidal PE?**

Sinusoidal PE adds positional information to the token embeddings before they enter the transformer. This means the positional signal gets mixed into the embedding through every subsequent linear projection — by the time you compute Q and K, the position information is entangled with the semantic content in a way that's hard for the model to cleanly separate.

RoPE applies the rotation directly to Q and K vectors just before the dot product. This keeps the positional information precisely where it matters (in the attention score computation) and out of where it doesn't (the value vectors). Crucially, it makes the dot product $q_m \cdot k_n^T$ a function of the relative position $(n - m)$ exactly, not approximately — which is the theoretical guarantee that makes RoPE superior.

```
  Sinusoidal: v_i* = v_i + p_i   →  position baked into embedding everywhere
  RoPE:       q_m  → q_m · R_m   →  position applied only at attention score step
              k_n  → k_n · R_n
              q_m · k_n^T = f(n - m)  exactly
```

RoPE also generalises better to sequence lengths not seen during training, which is why it is the default in modern models (LLaMA, Mistral, Gemma).

---

**Q: Why does ALiBi add a bias directly to attention *scores* rather than to input embeddings?**

Adding to embeddings changes what the token *is* before any processing happens. ALiBi's goal is narrower: just penalise attention scores between tokens that are far apart, so the model learns to rely more on nearby context.

By adding the bias directly to the pre-softmax attention scores — $\text{softmax}(q \cdot k^T / \sqrt{d_k} + \text{bias}(m,n))$ — ALiBi leaves the token representations completely untouched and targets exactly the right place: the weights that determine how much each token attends to every other. The bias is a simple linear function of distance, learned per head, and adds almost no parameters.

---
