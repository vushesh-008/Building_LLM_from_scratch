## 3. Full Encoder-Decoder Macro Flow

![Full encoder-decoder macro flow with cross-attention fanning to every decoder block](images/encoder_decoder_macro_flow_full.svg)

The purple bus line in the diagram is the detail most often missed: the final encoder output is **computed once** after all encoder blocks finish, then **broadcast identically** to the cross-attention sublayer of every decoder block.

```
  ENCODER STACK (left)           DECODER STACK (right)

  Source: "The cat"              Target: "<start> Le"
       │                               │
  Enc Blk 1                      Dec Blk 1 ◄───────────────────────┐
       │                               │                            │
  Enc Blk 2                      Dec Blk 2 ◄──────────────────┐    │
       │                               │                       │    │
      ...                             ...                      │    │
       │                               │                       │    │
  Enc Blk 6                      Dec Blk 6 ◄──────┐            │    │
       │                               │           │            │    │
  ┌────▼───────────────────────────────┘           │            │    │
  │  Final Encoder Output (K, V)   ────────────────┴────────────┴────┘
  └────────────────────────────────────────────────────────────────────
       computed ONCE — reused 6 times, one per decoder block
```

Decoder block 6's cross-attention reads the **exact same** K, V as decoder block 1's — neither has "more processed" access to the source sentence. The decoder self-attention stack gets progressively deeper representations of the target sequence; the encoder output it attends to never changes.

---
