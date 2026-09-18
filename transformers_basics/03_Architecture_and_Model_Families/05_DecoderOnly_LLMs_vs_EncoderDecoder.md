## 4. Decoder-Only LLMs vs Encoder-Decoder

When you read "GPT", "LLaMA", "Mistral", "Claude" — there is **no encoder stack** and **no cross-attention sublayer** anywhere. Each block has 2 sublayers, not 3.

![Decoder-only LLM block versus the original encoder-decoder decoder block](images/decoder_only_vs_encoder_decoder_block.svg)

| | Original decoder block | LLM decoder-only block |
|---|---|---|
| Sublayer 1 | Masked self-attention | Masked self-attention |
| Sublayer 2 | **Cross-attention** (needs encoder K, V) | ✗ removed entirely |
| Sublayer 3 | FFN | **FFN** (now sublayer 2) |
| Requires encoder? | Yes | No — encoder doesn't exist |
| Used in | Original Transformer, T5, BART | GPT, LLaMA, Mistral, Claude |

**What "N transformer blocks" means in an LLM paper:** $N$ copies of masked self-attention + Add&Norm + FFN + Add&Norm. That's the entire model, plus embedding lookup at the input and linear+softmax at the output.

```
  LLM block (repeated N times):

  Input from previous block
       │
  Masked Self-Attention   ← causal mask (can't see future tokens)
       │
  Add & Norm
       │
  FFN
       │
  Add & Norm
       │
  Output to next block
```

---
