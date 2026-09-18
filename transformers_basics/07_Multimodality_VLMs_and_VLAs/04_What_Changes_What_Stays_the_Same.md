## 2. What Changes, What Stays the Same

The decoder is **untouched**. You take a pretrained LLM and bolt a vision encoder onto the front. From the decoder's point of view, image tokens are just more entries in the input sequence — it has no idea they came from pixels instead of a vocabulary lookup.

```
  LLM:
  [Text tokens] → Causal Decoder → Next word

  VLM:
  [Image patches] → ViT Encoder → Adapter → ┐
                                              ├→ Causal Decoder → Next word
  [Text tokens]  → Embedding lookup      → ┘
```

| Component | LLM | VLM |
|---|---|---|
| Causal decoder | ✓ | ✓ unchanged |
| Token embedding lookup | ✓ | ✓ unchanged |
| Causal attention mask | ✓ | ✓ unchanged |
| Cross-entropy loss | ✓ | ✓ same formula |
| Vision encoder (ViT) | ✗ | ✓ added |
| Adapter / projection | ✗ | ✓ added |
| 2D positional encoding | ✗ | ✓ added (inside ViT) |
| Loss masking on image tokens | ✗ | ✓ added |

The additions are all **prefix** — they produce a sequence of vectors that get prepended to the text token sequence. Everything downstream (attention, FFN, output head) is unchanged.

---
