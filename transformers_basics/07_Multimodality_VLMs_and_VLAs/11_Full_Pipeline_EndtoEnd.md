## 9. Full Pipeline End-to-End

![LLM to VLM architectural pipeline](images/llm_to_vlm_pipeline.svg)

### The three-step transition: LLM → VLM → VLA

```
  LLM
  ├── Causal decoder                     ← kept
  ├── Text token embeddings (lookup)      ← kept
  └── CE loss on all text tokens          ← kept

  + ViT encoder (bidirectional attention on patches)
  + Patch embeddings (linear projection, not lookup)
  + Adapter (d_vit → d_model)
  + 2D positional encoding inside ViT
  + Loss mask (ignore image + prompt positions)
  = VLM

  + Widen vocabulary with action bin tokens
  + Robot state tokens (quantised joint angles)
  + Loss mask narrowed further (only action bins supervised)
  = VLA
```

> **The decoder never changes.** LLM → VLM → VLA is a series of *input side* and *vocabulary* extensions. The core transformer — attention, FFN, positional encoding, output head mechanics — is the same throughout.

---
