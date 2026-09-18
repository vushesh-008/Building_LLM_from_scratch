## 7. BERT Token-by-Token Routing

```mermaid
flowchart TD
    subgraph TOKENS ["Input Tokens"]
        direction LR
        T0["[CLS]\npos 0"] ~~~ T1["'The'\npos 1"] ~~~ T2["'dog'\npos 2"] ~~~ T3["[MASK]\npos 3"] ~~~ T4["[SEP]\npos 4"] ~~~ T5["'It'\npos 5"]
    end

    subgraph EMBEDS ["Input Vectors  (Token + Positional + Segment)"]
        direction LR
        E0["e₀"] ~~~ E1["e₁"] ~~~ E2["e₂"] ~~~ E3["e₃"] ~~~ E4["e₄"] ~~~ E5["e₅"]
    end

    subgraph BERT_LAYERS ["BERT Encoder Stack — Full Bidirectional Self-Attention"]
        direction LR
        BL1["Layer 1"] ~~~ BDOT["···"] ~~~ BLN["Layer N"]
    end

    subgraph FINAL ["Final Context Vectors"]
        direction LR
        H0["h₀\n[CLS]"] ~~~ H1["h₁\nignored"] ~~~ H2["h₂\nignored"] ~~~ H3["h₃\n[MASK]"] ~~~ H4["h₄\nignored"] ~~~ H5["h₅\nignored"]
    end

    subgraph NSP ["NSP Head  ←  h₀ [CLS]"]
        direction LR
        NL["Linear 768→2"] --> NS["Softmax"] --> NLoss["Binary CE Loss"]
    end

    subgraph MLM ["MLM Head  ←  h₃ [MASK]"]
        direction LR
        ML["Linear 768→30K"] --> MS["Softmax"] --> MLoss["CE Loss"]
    end

    TOKENS --> EMBEDS
    EMBEDS --> BERT_LAYERS
    BERT_LAYERS --> FINAL
    FINAL --> NSP
    FINAL --> MLM

    style H1 fill:#3a3a3a,color:#777777,stroke:#555555
    style H2 fill:#3a3a3a,color:#777777,stroke:#555555
    style H4 fill:#3a3a3a,color:#777777,stroke:#555555
    style H5 fill:#3a3a3a,color:#777777,stroke:#555555
    style NSP fill:#5c1e2e,color:#ffb3c1,stroke:#c44d6b
    style MLM fill:#1a2e5c,color:#a8c4ff,stroke:#3d68c8
```

**Key points:**
- Every token participates fully in bidirectional self-attention — context mixes across the entire sequence, left and right
- Only position 0 (`[CLS]`) feeds the NSP head; only masked positions feed the MLM head; all other positions are ignored at loss computation

---
