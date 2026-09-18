## 1. Output Head — From Hidden State to Probabilities

After the final decoder layer, each token position has a **context vector** $h \in \mathbb{R}^{d_{model}}$.
To generate the next token, two steps map this to a probability distribution over the vocabulary:

```mermaid
flowchart TD
    DEC["Last Decoder Layer"]
    H["h — Hidden State\nshape: d_model e.g. 768"]
    LIN["Linear  /  LM Head\nW ∈ [d_model × vocab_size]\nlogits = h · W\nshape: vocab_size e.g. 50,257"]
    SM["Softmax\nP_i = exp(logit_i) / Σ exp(logit_j)\nshape: vocab_size  — sums to 1.0"]
    OUT["Probability distribution over vocab\n'the' → 0.621\n'a'   → 0.153\n'an'  → 0.113\n'one' → 0.062\n'any' → 0.051\n...   → ...  (50K+ tokens)"]

    DEC --> H --> LIN --> SM --> OUT
```

> **Note:** The Linear weight matrix $W$ is often **tied** to the input token embedding matrix — the same weights used to embed tokens at the input are transposed and reused here. This saves parameters and often improves performance.

---
