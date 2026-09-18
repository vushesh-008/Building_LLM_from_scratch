## 8. BERT Loss & Word Embeddings

$$L_{\text{total}} = L_{\text{MLM}} + L_{\text{NSP}}$$

**NSP Loss (position 0 only):**

$$L_{\text{NSP}} = -\log P(\text{IsNext} \mid h_{\text{CLS}})$$

**MLM Loss (masked positions only):**

$$L_{\text{MLM}} = -\log P(\text{true token} \mid h_{\text{mask}})$$

**Concrete example:**

$$L_{\text{NSP}} = -\log(0.85) \approx 0.16 \qquad L_{\text{MLM}} = -\log(0.70) \approx 0.36$$

$$L_{\text{total}} \approx 0.52$$

A single `L_total.backward()` computes all gradients in one pass.

### Are word embeddings learned or pre-set?

**Before pre-training:** the embedding table is **randomly initialised** — no Word2Vec, no prior embeddings, nothing.

**During pre-training:** the embedding table rows are updated by backpropagation exactly like any weight matrix. By the end of pre-training, semantically similar words sit close together in embedding space.

$$e_0, e_1, \ldots, e_n \quad \text{(input token embeddings)} \quad \xrightarrow{\text{N encoder layers}} \quad h_0, h_1, \ldots, h_n \quad \text{(contextual representations)}$$

---
