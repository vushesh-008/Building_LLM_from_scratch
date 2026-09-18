## 2. The Original Transformer Architecture

The 2017 "Attention is All You Need" model consists of two separate stacks — an Encoder and a Decoder — each containing $N=6$ identical blocks.

| Component | Encoder | Decoder |
|---|---|---|
| Input | Token embedding + Positional encoding | Output embedding + Positional encoding (shifted right) |
| Attention | Multi-Head **Self**-Attention (bidirectional) | Masked Multi-Head Self-Attention + **Cross**-Attention |
| Other | Add & Norm, FFN | Add & Norm, FFN |
| Output | Contextual representations | Token probabilities via Linear + Softmax |
| Repeated | N times (N=6) | N times (same N) |

### Cross-Attention

Cross-attention is where **Q comes from one sequence and K, V come from another**:

```
  Self-attention:
  Q = X · W_Q  ┐
  K = X · W_K  ├── all from the same input X
  V = X · W_V  ┘

  Cross-attention (inside each decoder block):
  Q = X_decoder · W_Q   ← from the decoder  ("what am I looking for?")
  K = X_encoder · W_K   ← from the encoder  ("what do I have?")
  V = X_encoder · W_V   ← from the encoder  ("what do I return?")
```

### Translation example

```
  Translating "The cat sat" → "Le chat s'est assis"

  Encoder processes: "The cat sat" → hidden states [h₁, h₂, h₃]

  Decoder generating "chat":
    Q = embedding("Le") · W_Q      ← decoder's current query
    K = [h₁, h₂, h₃] · W_K       ← full encoder output
    V = [h₁, h₂, h₃] · W_V
    → attention peaks at h₂ ("cat") ← alignment learned automatically
```

| | Self-Attention | Cross-Attention |
|---|---|---|
| Q source | Same sequence | Decoder |
| K, V source | Same sequence | Encoder |
| Purpose | Attend within one sequence | Attend across two sequences |
| Used in | All transformers | Encoder-decoder only |

---
