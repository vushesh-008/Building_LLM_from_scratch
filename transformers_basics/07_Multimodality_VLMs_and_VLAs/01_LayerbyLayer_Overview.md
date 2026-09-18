## Layer-by-Layer Overview

A VLM is a pretrained LLM with three components bolted onto the front. From input pixel to output word:

| Layer | What it does | Key detail |
|---|---|---|
| **Patch embedding** | Splits the image into fixed-size patches; each patch is flattened and linearly projected to a vector | Computation (matrix multiply), not a lookup |
| **Positional encoding (ViT)** | Adds a learned position vector to each patch embedding so the ViT knows where each patch is in the grid | Standard practice: 1D learned, not 2D |
| **ViT encoder** | N transformer encoder blocks with **bidirectional** (unmasked) self-attention — every patch attends to every other | Produces rich per-patch representations; also outputs a [CLS] token for global image summary |
| **Adapter / projection** | Maps ViT output dimension ($d_{vit}$) to LLM input dimension ($d_{model}$) | Linear, MLP, Q-Former, or Perceiver |
| **LLM decoder** | Standard causal transformer — image tokens prepended as a prefix, then text tokens follow | Unchanged from the base LLM |
| **Output head** | Linear + softmax over vocabulary | Identical to plain LLM; loss computed only on response tokens |

---
